# =============================================================================
# CEO Assistant — src/core/agent.py
# Orquestrador LangGraph — Grafo de Estados do Agente
#
# Por que LangGraph em vez de LangChain Chains simples?
# - Controle explícito de fluxo: cada etapa é um nó nomeado no grafo
# - Facilita debugging: é possível inspecionar o estado após cada nó
# - Conditional edges: roteamento baseado em condições (ex: skill identificada)
# - Resiliência: cada nó pode ter retry independente sem afetar os demais
# - Observabilidade: o grafo visualizável facilita entender o fluxo de dados
#
# Grafo de estados:
#
#  START
#    │
#    ▼
# [validate_message]
#    │
#    ▼
# [interpret_intent]  ←── extrai período, entidade, tipo de dado
#    │
#    ▼
# [route_skill]  ←── decide qual skill usar (financeiro/vendas/churn/geral)
#    │
#    ├── financeiro ──► [execute_financeiro]
#    ├── vendas ──────► [execute_vendas]
#    ├── churn ───────► [execute_churn]
#    └── geral/unknown► [execute_geral]
#         │
#         ▼
#    [format_response]  ←── LLM formata a resposta final com dados
#         │
#         ▼
#    [send_message]  ←── Evolution API /message/sendText
#         │
#         ▼
#    [log_interaction]  ←── SQLite
#         │
#         ▼
#        END
# =============================================================================

import time
import structlog
from typing import Any, Literal
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START

from src.core.llm import get_llm, get_creative_llm
from src.core.router import route_skill, SkillName, RouteResult

logger = structlog.get_logger(__name__)


# =============================================================================
# ESTADO DO AGENTE
# O estado é o objeto que flui pelo grafo — cada nó pode ler e escrever nele.
# Usamos dataclass para type hints claros e imutabilidade controlada.
# =============================================================================

@dataclass
class AgentState:
    """Estado completo que flui pelo grafo LangGraph."""

    # --- Entrada ---
    query: str = ""                          # pergunta original do CEO
    remote_jid: str = ""                     # número WhatsApp do remetente

    # --- Interpretação ---
    intent: dict[str, Any] = field(default_factory=dict)
    # Ex: {"periodo": "julho/2025", "entidade": "faturamento", "tipo": "consulta"}

    # --- Roteamento ---
    route: RouteResult | None = None         # resultado do roteador de skills

    # --- Execução da skill ---
    skill_data: dict[str, Any] = field(default_factory=dict)
    # Dados brutos retornados pela skill (dict com valores, fonte, período)

    # --- Resposta ---
    formatted_response: str = ""             # texto formatado para o WhatsApp

    # --- Controle de fluxo ---
    error: str | None = None                 # mensagem de erro se algo falhar
    should_respond: bool = True             # False = mensagem ignorada (bot, etc.)

    # --- Metadados para logging ---
    start_time: float = field(default_factory=time.time)
    skill_used: str = ""
    data_source: str = ""


# =============================================================================
# PROMPT PARA INTERPRETAÇÃO DE INTENÇÃO
# =============================================================================

_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Você analisa perguntas executivas e extrai informações estruturadas.
Responda em JSON com os campos abaixo (sem markdown, JSON puro):

{
  "periodo": "período mencionado (ex: julho/2025, Q3 2025, últimos 30 dias, YTD) ou null",
  "entidade": "o que está sendo consultado (ex: faturamento, receita, churn, vendas) ou null",
  "tipo": "tipo de consulta: comparativo | pontual | tendência | ranking | resumo",
  "filtros": "filtros adicionais mencionados (ex: região sul, produto X) ou null"
}"""
    ),
    ("human", "{query}"),
])

# =============================================================================
# PROMPT PARA FORMATAÇÃO DA RESPOSTA FINAL
# =============================================================================

_FORMAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Você é o assistente executivo pessoal do CEO. Sua missão é responder perguntas
sobre dados do negócio de forma clara, precisa e executiva.

Regras:
- Responda em português do Brasil
- Use emojis com moderação (apenas para destacar tendências: 🟢🔴📈📉)
- Formate valores monetários em R$ com separador de milhar (ex: R$ 2.847.320,00)
- Formate porcentagens com 1 casa decimal (ex: +9,1%)
- Sempre cite a fonte dos dados e o período
- Seja conciso: máximo 5 linhas, exceto se for um resumo completo
- Use *negrito* para WhatsApp (asteriscos simples)
- Se os dados forem simulados/fictícios, NÃO mencione isso

Dados disponíveis:
{skill_data}

Intenção do CEO:
{intent}"""
    ),
    ("human", "{query}"),
])


# =============================================================================
# NÓS DO GRAFO
# Cada nó recebe o estado atual e retorna um dict com as chaves a atualizar.
# =============================================================================

def node_validate_message(state: AgentState) -> dict:
    """
    Nó 1: Valida se a mensagem deve ser processada.
    Filtra mensagens de outros bots, grupos, status, etc.
    """
    query = state.query.strip()

    # Ignora mensagens vazias
    if not query:
        logger.info("Mensagem vazia ignorada", jid=state.remote_jid)
        return {"should_respond": False, "error": "Mensagem vazia"}

    # Ignora mensagens muito curtas (ex: "ok", "sim") — abaixo de 3 chars
    if len(query) < 3:
        return {"should_respond": False, "error": "Mensagem muito curta"}

    logger.info(
        "Mensagem válida recebida",
        jid=state.remote_jid,
        query_preview=query[:50],
    )
    return {"should_respond": True}


def node_interpret_intent(state: AgentState) -> dict:
    """
    Nó 2: Usa o LLM para extrair informações estruturadas da pergunta.
    Identifica período, entidade consultada e tipo de consulta.
    """
    import json

    llm = get_llm()
    chain = _INTENT_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({"query": state.query})
        # Remove possíveis blocos de código markdown
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        intent = json.loads(raw)
        logger.debug("Intenção extraída", intent=intent)
        return {"intent": intent}

    except Exception as exc:
        logger.warning("Falha ao extrair intenção, usando fallback", error=str(exc))
        # Fallback: intenção mínima para não travar o fluxo
        return {
            "intent": {
                "periodo": None,
                "entidade": None,
                "tipo": "pontual",
                "filtros": None,
            }
        }


def node_route_skill(state: AgentState) -> dict:
    """
    Nó 3: Roteia para a skill correta baseado na pergunta.
    """
    route = route_skill(state.query)
    logger.info(
        "Skill selecionada",
        skill=route.skill.value,
        confidence=route.confidence,
        method=route.method,
    )
    return {"route": route, "skill_used": route.skill.value}


def node_execute_financeiro(state: AgentState) -> dict:
    """
    Nó 4a: Executa a skill financeira.
    """
    # Import aqui para evitar circular imports e para carregamento lazy
    from src.skills.skill_financeiro import SkillFinanceiro

    skill = SkillFinanceiro()
    data = skill.execute(
        query=state.query,
        intent=state.intent,
    )
    return {"skill_data": data, "data_source": data.get("source", "financeiro")}


def node_execute_vendas(state: AgentState) -> dict:
    """
    Nó 4b: Executa a skill de vendas.
    (Implementação completa em etapa futura — retorna placeholder por ora)
    """
    logger.info("Skill vendas acionada (placeholder)", query=state.query[:50])
    return {
        "skill_data": {
            "message": "Skill de vendas em desenvolvimento.",
            "source": "vendas",
            "periodo": state.intent.get("periodo", "não especificado"),
        }
    }


def node_execute_churn(state: AgentState) -> dict:
    """
    Nó 4c: Executa a skill de churn.
    (Implementação completa em etapa futura — retorna placeholder por ora)
    """
    logger.info("Skill churn acionada (placeholder)", query=state.query[:50])
    return {
        "skill_data": {
            "message": "Skill de churn em desenvolvimento.",
            "source": "churn",
            "periodo": state.intent.get("periodo", "não especificado"),
        }
    }


def node_execute_geral(state: AgentState) -> dict:
    """
    Nó 4d: Skill geral — para perguntas que abrangem múltiplas áreas.
    Usa RAG para buscar contexto relevante nos documentos indexados.
    """
    from src.core.rag import get_rag_retriever

    logger.info("Skill geral (RAG) acionada", query=state.query[:50])
    try:
        retriever = get_rag_retriever()
        docs = retriever.get_relevant_documents(state.query)
        context = "\n\n".join([d.page_content for d in docs[:3]])

        return {
            "skill_data": {
                "context": context,
                "source": "base de conhecimento (RAG)",
                "periodo": "geral",
            }
        }
    except Exception as exc:
        logger.error("Falha no RAG, usando resposta genérica", error=str(exc))
        return {
            "skill_data": {
                "message": "Não foi possível recuperar informações específicas sobre essa consulta.",
                "source": "sem dados",
                "periodo": "N/A",
            }
        }


def node_format_response(state: AgentState) -> dict:
    """
    Nó 5: LLM formata os dados brutos em resposta executiva para WhatsApp.
    """
    import json

    llm = get_creative_llm()
    chain = _FORMAT_PROMPT | llm | StrOutputParser()

    try:
        skill_data_str = json.dumps(state.skill_data, ensure_ascii=False, indent=2)
        intent_str = json.dumps(state.intent, ensure_ascii=False)

        response = chain.invoke({
            "query": state.query,
            "skill_data": skill_data_str,
            "intent": intent_str,
        })

        logger.debug("Resposta formatada", preview=response[:100])
        return {"formatted_response": response.strip()}

    except Exception as exc:
        logger.error("Falha ao formatar resposta", error=str(exc))
        return {
            "formatted_response": (
                "❌ Desculpe, não consegui processar sua consulta agora. "
                "Tente novamente em instantes."
            )
        }


def node_send_message(state: AgentState) -> dict:
    """
    Nó 6: Envia a resposta via Evolution API.
    Import lazy para evitar dependência circular e facilitar testes unitários.
    """
    from src.api.evolution import send_text_message
    import asyncio

    try:
        # Como estamos em contexto síncrono do LangGraph, usamos asyncio.run
        # Em produção com FastAPI, use um event loop já existente
        asyncio.run(
            send_text_message(
                to=state.remote_jid,
                text=state.formatted_response,
            )
        )
        logger.info(
            "Mensagem enviada com sucesso",
            to=state.remote_jid,
            preview=state.formatted_response[:50],
        )
    except Exception as exc:
        logger.error("Falha ao enviar mensagem", error=str(exc))

    return {}  # Não altera estado


def node_log_interaction(state: AgentState) -> dict:
    """
    Nó 7: Registra a interação completa no SQLite para auditoria.
    """
    import asyncio
    from src.utils.db import log_interaction

    elapsed = time.time() - state.start_time

    try:
        asyncio.run(
            log_interaction(
                remote_jid=state.remote_jid,
                query=state.query,
                response=state.formatted_response,
                skill=state.skill_used,
                data_source=state.data_source,
                elapsed_seconds=elapsed,
            )
        )
        logger.info(
            "Interação registrada",
            skill=state.skill_used,
            elapsed_ms=round(elapsed * 1000),
        )
    except Exception as exc:
        logger.error("Falha ao registrar interação no SQLite", error=str(exc))

    return {}


# =============================================================================
# CONDIÇÕES DE ROTEAMENTO (Conditional Edges)
# =============================================================================

def should_continue(state: AgentState) -> Literal["interpret_intent", "end"]:
    """Verifica se a mensagem deve ser processada após validação."""
    if not state.should_respond:
        return "end"
    return "interpret_intent"


def select_skill_node(
    state: AgentState,
) -> Literal["execute_financeiro", "execute_vendas", "execute_churn", "execute_geral"]:
    """Seleciona qual nó de skill executar baseado no resultado do roteador."""
    if state.route is None:
        return "execute_geral"

    skill_map = {
        SkillName.FINANCEIRO: "execute_financeiro",
        SkillName.VENDAS: "execute_vendas",
        SkillName.CHURN: "execute_churn",
        SkillName.GERAL: "execute_geral",
        SkillName.DESCONHECIDO: "execute_geral",
    }
    return skill_map.get(state.route.skill, "execute_geral")


# =============================================================================
# CONSTRUÇÃO DO GRAFO
# =============================================================================

def build_agent_graph() -> StateGraph:
    """
    Constrói e compila o grafo LangGraph do agente CEO Assistant.

    Returns:
        Grafo compilado pronto para execução.
    """
    # Cria o grafo com o tipo de estado definido
    graph = StateGraph(AgentState)

    # --- Adiciona os nós ---
    graph.add_node("validate_message", node_validate_message)
    graph.add_node("interpret_intent", node_interpret_intent)
    graph.add_node("route_skill", node_route_skill)
    graph.add_node("execute_financeiro", node_execute_financeiro)
    graph.add_node("execute_vendas", node_execute_vendas)
    graph.add_node("execute_churn", node_execute_churn)
    graph.add_node("execute_geral", node_execute_geral)
    graph.add_node("format_response", node_format_response)
    graph.add_node("send_message", node_send_message)
    graph.add_node("log_interaction", node_log_interaction)

    # --- Define as arestas ---
    # Ponto de entrada
    graph.add_edge(START, "validate_message")

    # Conditional edge: após validação, decide se continua ou termina
    graph.add_conditional_edges(
        "validate_message",
        should_continue,
        {
            "interpret_intent": "interpret_intent",
            "end": END,
        },
    )

    # Fluxo linear até o roteador
    graph.add_edge("interpret_intent", "route_skill")

    # Conditional edge: após roteamento, vai para a skill correta
    graph.add_conditional_edges(
        "route_skill",
        select_skill_node,
        {
            "execute_financeiro": "execute_financeiro",
            "execute_vendas": "execute_vendas",
            "execute_churn": "execute_churn",
            "execute_geral": "execute_geral",
        },
    )

    # Todas as skills convergem para format_response
    graph.add_edge("execute_financeiro", "format_response")
    graph.add_edge("execute_vendas", "format_response")
    graph.add_edge("execute_churn", "format_response")
    graph.add_edge("execute_geral", "format_response")

    # Fluxo final
    graph.add_edge("format_response", "send_message")
    graph.add_edge("send_message", "log_interaction")
    graph.add_edge("log_interaction", END)

    return graph.compile()


# Instância singleton do grafo compilado
_agent_graph = None


def get_agent() -> StateGraph:
    """
    Retorna o grafo do agente (singleton — compilado apenas uma vez na startup).
    """
    global _agent_graph
    if _agent_graph is None:
        logger.info("Compilando grafo do agente LangGraph...")
        _agent_graph = build_agent_graph()
        logger.info("Grafo compilado com sucesso")
    return _agent_graph


async def run_agent(query: str, remote_jid: str) -> str:
    """
    Ponto de entrada principal para processar uma mensagem do CEO.

    Args:
        query: Texto da mensagem recebida.
        remote_jid: Identificador WhatsApp do remetente.

    Returns:
        Resposta formatada (também enviada via Evolution API internamente).
    """
    agent = get_agent()

    initial_state = AgentState(
        query=query,
        remote_jid=remote_jid,
        start_time=time.time(),
    )

    logger.info(
        "Iniciando processamento do agente",
        jid=remote_jid,
        query_preview=query[:60],
    )

    # Executa o grafo de forma síncrona (LangGraph suporta execução sync)
    final_state = await agent.ainvoke(initial_state)

    return final_state.get("formatted_response", "")
