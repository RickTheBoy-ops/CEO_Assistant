# =============================================================================
# CEO Assistant — src/core/rag.py
# Camada RAG: Indexação e Recuperação por Similaridade Semântica
#
# Por que RAG aqui?
# Perguntas gerais do CEO ("como estamos indo?", "qual o resumo do Q3?")
# precisam de contexto que não cabe num prompt fixo. O RAG permite indexar
# documentos variados (relatórios, notas de reunião, metas) e recuperar
# apenas os trechos mais relevantes para a pergunta, mantendo o prompt
# dentro do limite de tokens do LLM.
#
# Por que ChromaDB?
# - Self-hosted: dados ficam na sua máquina, sem enviar para API externa
# - Persistência: banco salvo em disco (volume Docker), sobrevive a reinícios
# - Simples: API Python nativa, sem infraestrutura adicional
# - Escalável: suporta milhões de embeddings em instância local
#
# Por que sentence-transformers (all-MiniLM-L6-v2)?
# - Roda localmente sem GPU
# - 384 dimensões: ótimo equilíbrio velocidade/qualidade
# - Open-source e amplamente adotado
# - Pode ser substituído por embeddings OpenAI via variável de config
# =============================================================================

import hashlib
import structlog
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import get_settings

logger = structlog.get_logger(__name__)

# Nome da coleção principal no ChromaDB
COLLECTION_NAME = "ceo_assistant_knowledge"


# =============================================================================
# DOCUMENTOS INICIAIS (seed data)
# Em produção, esses documentos seriam importados de Google Sheets,
# relatórios PDF, notas de reunião, etc.
# =============================================================================

SEED_DOCUMENTS: list[dict[str, Any]] = [
    {
        "content": """Relatório Executivo Q2 2025 — Resumo Financeiro
Faturamento Total: R$ 7.811.190,00 (Abril + Maio + Junho)
Crescimento vs Q1 2025: +4,2%
Margem Bruta Média: 29,9%
Destaques: Junho foi abaixo da média por sazonalidade (recesso escolar).
Meta Q2: R$ 7.500.000,00 — META ATINGIDA (+4,1%)""",
        "metadata": {"categoria": "financeiro", "periodo": "Q2 2025", "tipo": "relatorio"},
    },
    {
        "content": """Metas Estratégicas 2025
- Faturamento anual: R$ 33.000.000,00 (crescimento 8% vs 2024)
- Margem bruta: manter acima de 28%
- NPS: atingir 72 pontos (atual: 67)
- Churn mensal: reduzir para abaixo de 1,8% (atual: 2,1%)
- Novos clientes: 48 no ano (4 por mês em média)
- Expansão: entrar no mercado do Nordeste no Q3""",
        "metadata": {"categoria": "estrategia", "periodo": "2025", "tipo": "metas"},
    },
    {
        "content": """Status de Clientes — Julho 2025
Total de clientes ativos: 312
Novos clientes no mês: 8
Cancelamentos no mês: 7 (churn: 2,24%)
Clientes em risco (NPS < 30): 18
Renovações no mês: 14 contratos — todas realizadas
Ticket médio mensal: R$ 9.126,67
Maior cliente: Grupo Fortuna Atacadista (R$ 48.000/mês)""",
        "metadata": {"categoria": "clientes", "periodo": "Julho 2025", "tipo": "status"},
    },
    {
        "content": """Pipeline de Vendas — Julho 2025
Propostas em andamento: 23 (valor total: R$ 1.840.000,00 ARR)
Stage 1 (Descoberta): 8 propostas — R$ 480.000 ARR
Stage 2 (Demo feita): 9 propostas — R$ 810.000 ARR
Stage 3 (Proposta enviada): 6 propostas — R$ 550.000 ARR
Taxa de conversão histórica: 31% das propostas se tornam clientes
Previsão de fechamento em julho: 3-4 contratos (R$ 165.000-220.000 ARR)""",
        "metadata": {"categoria": "vendas", "periodo": "Julho 2025", "tipo": "pipeline"},
    },
    {
        "content": """Indicadores Operacionais — Junho 2025
SLA de Suporte: 94,2% (meta: 95%) — ligeiramente abaixo
Tempo médio de resolução: 4,2 horas (meta: 4h)
Uptime da plataforma: 99,94%
Tickets abertos: 47 (volume normal)
Satisfação CSAT: 4,3/5,0
Time: 67 colaboradores (3 vagas abertas em Tech)""",
        "metadata": {"categoria": "operacional", "periodo": "Junho 2025", "tipo": "kpi"},
    },
    {
        "content": """Análise de Churn — Primeiro Semestre 2025
Churn total H1: 38 cancelamentos
Principal motivo: insatisfação com suporte (34%)
Segundo motivo: redução de budget do cliente (28%)
Terceiro motivo: migração para concorrente (21%)
Churn por porte: PME (55%), Médias empresas (32%), Enterprise (13%)
Ações em andamento: programa de customer success ativo para clientes com NPS < 40""",
        "metadata": {"categoria": "churn", "periodo": "H1 2025", "tipo": "analise"},
    },
    {
        "content": """Resumo Executivo — Reunião de Liderança 15/07/2025
Temas discutidos:
1. Faturamento de junho abaixo do esperado — aprovado plano de aceleração em agosto
2. Contratação de 2 Customer Success Managers até agosto
3. Lançamento da feature de integração com ERP previsto para Q4 2025
4. Abertura de escritório em Recife confirmada para setembro
5. Próxima revisão de preços: reajuste de 8% para novos contratos em outubro
Decisões aprovadas: plano de CS, expansão Nordeste, revisão de preços Q4""",
        "metadata": {"categoria": "reuniao", "periodo": "Julho 2025", "tipo": "ata"},
    },
]


# =============================================================================
# EMBEDDINGS
# =============================================================================

def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Retorna o modelo de embeddings local.

    Por que all-MiniLM-L6-v2?
    - 22MB de tamanho: download rápido no primeiro uso
    - 384 dimensões: processamento rápido mesmo em CPU
    - Score MTEB aceitável para buscas em português (funciona bem com texto bilíngue)
    - Alternativa gratuita aos embeddings da OpenAI
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# =============================================================================
# INDEXER
# =============================================================================

class RAGIndexer:
    """
    Responsável por indexar documentos no ChromaDB.

    Uso:
        indexer = RAGIndexer()
        indexer.index_seed_documents()                    # indexa dados iniciais
        indexer.add_document("texto...", {"fonte": "..."})  # adiciona documento avulso
    """

    def __init__(self) -> None:
        settings = get_settings()
        persist_dir = settings.chroma_persist_dir

        # Cria diretório se não existir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._embeddings = _get_embeddings()
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
        )
        logger.info("RAGIndexer inicializado", persist_dir=persist_dir)

    def index_seed_documents(self, force: bool = False) -> int:
        """
        Indexa os documentos iniciais (SEED_DOCUMENTS).

        Args:
            force: Se True, re-indexa mesmo que já existam documentos.

        Returns:
            Número de documentos indexados nesta chamada.
        """
        # Verifica se já foram indexados
        existing_count = self._vectorstore._collection.count()
        if existing_count > 0 and not force:
            logger.info(
                "Documentos seed já indexados, pulando",
                count=existing_count,
            )
            return 0

        documents = []
        ids = []
        metadatas = []
        texts = []

        for doc in SEED_DOCUMENTS:
            content = doc["content"]
            metadata = doc["metadata"]

            # Gera ID determinístico baseado no conteúdo (evita duplicatas)
            doc_id = hashlib.md5(content.encode()).hexdigest()

            documents.append(Document(page_content=content, metadata=metadata))
            ids.append(doc_id)
            metadatas.append(metadata)
            texts.append(content)

        self._vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info("Documentos seed indexados", count=len(documents))
        return len(documents)

    def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """
        Adiciona um documento avulso ao índice.

        Args:
            content: Conteúdo textual do documento.
            metadata: Metadados opcionais (categoria, período, fonte, etc.).
            doc_id: ID personalizado. Se None, gera hash MD5 do conteúdo.

        Returns:
            ID do documento indexado.
        """
        if doc_id is None:
            doc_id = hashlib.md5(content.encode()).hexdigest()

        self._vectorstore.add_texts(
            texts=[content],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )

        logger.info(
            "Documento indexado",
            doc_id=doc_id,
            content_preview=content[:50],
        )
        return doc_id

    def get_document_count(self) -> int:
        """Retorna o total de documentos indexados."""
        return self._vectorstore._collection.count()


# =============================================================================
# RETRIEVER
# =============================================================================

class RAGRetriever:
    """
    Responsável por recuperar documentos relevantes por similaridade semântica.

    Uso:
        retriever = RAGRetriever()
        docs = retriever.search("qual o status do churn em julho?", k=3)
        for doc in docs:
            print(doc.page_content, doc.metadata)
    """

    def __init__(self) -> None:
        settings = get_settings()
        persist_dir = settings.chroma_persist_dir

        self._embeddings = _get_embeddings()
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
        )
        logger.info("RAGRetriever inicializado", persist_dir=persist_dir)

    def search(
        self,
        query: str,
        k: int = 3,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Busca os k documentos mais relevantes para a query.

        Args:
            query: Pergunta ou texto de busca.
            k: Número de documentos a retornar.
            filter_metadata: Filtro opcional por campos de metadata
                             Ex: {"categoria": "financeiro"}

        Returns:
            Lista de Documents ordenados por similaridade (maior primeiro).
        """
        try:
            docs = self._vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_metadata,
            )
            logger.debug(
                "RAG: documentos recuperados",
                query_preview=query[:50],
                n_docs=len(docs),
            )
            return docs

        except Exception as exc:
            logger.error("Falha na busca RAG", error=str(exc))
            return []

    def search_with_scores(
        self,
        query: str,
        k: int = 3,
    ) -> list[tuple[Document, float]]:
        """
        Busca documentos com score de similaridade (útil para debugging).

        Returns:
            Lista de tuplas (Document, score) onde score ∈ [0, 1].
        """
        try:
            return self._vectorstore.similarity_search_with_score(query=query, k=k)
        except Exception as exc:
            logger.error("Falha na busca RAG com scores", error=str(exc))
            return []

    def get_relevant_documents(self, query: str) -> list[Document]:
        """
        Interface compatível com LangChain BaseRetriever.
        Usado pelo nó execute_geral do agente.
        """
        return self.search(query=query, k=3)


# =============================================================================
# SINGLETONS
# =============================================================================

_indexer: RAGIndexer | None = None
_retriever: RAGRetriever | None = None


def get_rag_indexer() -> RAGIndexer:
    """Retorna instância singleton do RAGIndexer."""
    global _indexer
    if _indexer is None:
        _indexer = RAGIndexer()
    return _indexer


def get_rag_retriever() -> RAGRetriever:
    """Retorna instância singleton do RAGRetriever."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


def initialize_rag() -> None:
    """
    Inicializa o RAG na startup da aplicação:
    - Cria o indexer e o retriever
    - Indexa os documentos seed se ainda não foram indexados

    Chamado no startup event do FastAPI em main.py.
    """
    logger.info("Inicializando camada RAG...")
    indexer = get_rag_indexer()
    indexed = indexer.index_seed_documents()
    total = indexer.get_document_count()
    logger.info(
        "RAG inicializado",
        novos_indexados=indexed,
        total_documentos=total,
    )
    # Pré-carrega o retriever também
    get_rag_retriever()
