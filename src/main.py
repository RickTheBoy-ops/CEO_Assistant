# =============================================================================
# CEO Assistant — src/main.py
# Ponto de Entrada da Aplicação FastAPI
#
# Responsabilidades do main.py:
# - Configurar logging estruturado
# - Inicializar o banco SQLite e o ChromaDB RAG no startup
# - Registrar todos os routers FastAPI
# - Expor endpoints utilitários: /health, /status, /stats
# =============================================================================

import asyncio
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.logger import setup_logging
from src.utils.db import initialize_db, get_stats
from src.api.webhook import router as webhook_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.

    Por que usar lifespan em vez de @app.on_event("startup")?
    - lifespan é o padrão moderno do FastAPI (on_event está deprecated)
    - O contexto async garante cleanup adequado mesmo em caso de erro
    - Código antes do 'yield' = startup; código após = shutdown
    """
    # -------------------------------------------------------------------------
    # STARTUP
    # -------------------------------------------------------------------------

    # 1. Configura logging estruturado primeiro (outros módulos dependem disso)
    setup_logging()
    logger.info("🚀 CEO Assistant iniciando...")

    # 2. Inicializa banco SQLite de logs
    logger.info("Inicializando banco SQLite...")
    await initialize_db()

    # 3. Inicializa RAG (ChromaDB + embeddings) — pode demorar no primeiro boot
    # (download do modelo sentence-transformers ~22MB)
    logger.info("Inicializando RAG (ChromaDB + embeddings)...")
    try:
        from src.core.rag import initialize_rag
        await asyncio.get_event_loop().run_in_executor(None, initialize_rag)
        logger.info("RAG inicializado com sucesso ✓")
    except Exception as exc:
        logger.error("Falha ao inicializar RAG", error=str(exc))
        # Não bloqueia o startup — app funciona sem RAG (skills diretas ainda funcionam)

    # 4. Compila o grafo LangGraph (evita delay na primeira mensagem)
    logger.info("Pré-compilando grafo LangGraph...")
    try:
        from src.core.agent import get_agent
        get_agent()
        logger.info("Grafo LangGraph compilado ✓")
    except Exception as exc:
        logger.error("Falha ao compilar grafo LangGraph", error=str(exc))

    logger.info("✅ CEO Assistant pronto para receber mensagens")

    yield  # Aplicação em execução

    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------
    logger.info("CEO Assistant encerrando...")


# =============================================================================
# INICIALIZAÇÃO DA APLICAÇÃO
# =============================================================================

app = FastAPI(
    title="CEO Assistant",
    description=(
        "Assistente executivo pessoal via WhatsApp. "
        "Consulte KPIs e dashboards em tempo real usando linguagem natural."
    ),
    version="1.0.0",
    docs_url="/docs",        # Swagger UI (útil para debug em dev)
    redoc_url="/redoc",      # ReDoc
    lifespan=lifespan,
)

# CORS — necessário se você tiver um frontend admin acessando a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Em produção, restrinja aos seus domínios
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Registra os routers
app.include_router(webhook_router)


# =============================================================================
# ENDPOINTS UTILITÁRIOS
# =============================================================================

@app.get(
    "/health",
    tags=["Monitoramento"],
    summary="Healthcheck básico",
)
async def health_check() -> dict:
    """
    Endpoint de healthcheck — usado pelo Docker e load balancers.
    Retorna 200 OK se a aplicação está em pé.
    """
    return {"status": "ok", "service": "CEO Assistant", "version": "1.0.0"}


@app.get(
    "/status",
    tags=["Monitoramento"],
    summary="Status completo da aplicação",
)
async def full_status() -> dict:
    """
    Verifica o status de todos os componentes:
    - Conexão WhatsApp (Evolution API)
    - ChromaDB (RAG)
    - SQLite (logs)
    """
    from src.api.evolution import check_connection

    whatsapp_ok = False
    try:
        whatsapp_ok = await check_connection()
    except Exception:
        pass

    rag_ok = False
    rag_docs = 0
    try:
        from src.core.rag import get_rag_indexer
        indexer = get_rag_indexer()
        rag_docs = indexer.get_document_count()
        rag_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "components": {
            "whatsapp": {"connected": whatsapp_ok},
            "rag": {"ok": rag_ok, "document_count": rag_docs},
            "sqlite": {"ok": True},
        },
    }


@app.get(
    "/stats",
    tags=["Monitoramento"],
    summary="Estatísticas de uso do assistente",
)
async def usage_stats() -> dict:
    """
    Retorna estatísticas de uso registradas no SQLite:
    - Total de interações
    - Tempo médio de resposta
    - Breakdown de uso por skill
    """
    return await get_stats()
