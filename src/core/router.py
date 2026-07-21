# =============================================================================
# CEO Assistant — src/core/router.py
# Roteador de Skills por Intenção
#
# Por que duas camadas de roteamento (keyword + LLM)?
# - Keywords são rápidas, determinísticas e gratuitas (sem chamada LLM)
# - O fallback LLM garante que perguntas ambíguas ou mal formuladas ainda
#   cheguem à skill correta (ex: "como estamos indo?" → precisa LLM pra saber)
# - Essa abordagem é chamada de "hybrid routing" e é comum em sistemas de
#   produção onde latência e custo importam.
# =============================================================================

import re
import structlog
from enum import Enum
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.llm import get_llm

logger = structlog.get_logger(__name__)


class SkillName(str, Enum):
    """Skills disponíveis no sistema."""
    FINANCEIRO = "financeiro"   # Faturamento, receita, custos, margem
    VENDAS = "vendas"           # Volume, ticket médio, conversão, pipeline
    CHURN = "churn"             # Cancelamentos, LTV, clientes em risco
    GERAL = "geral"             # Fallback: perguntas gerais sobre o negócio
    DESCONHECIDO = "desconhecido"  # Não foi possível identificar a intenção


@dataclass
class RouteResult:
    """Resultado do roteamento com skill identificada e confiança."""
    skill: SkillName
    confidence: str          # "high" (keyword match) | "medium" (LLM) | "low" (fallback)
    method: str              # "keyword" | "llm"
    original_query: str


# ---------------------------------------------------------------------------
# Dicionário de keywords por skill
# Organizado do mais específico para o mais genérico para evitar falsos positivos
# ---------------------------------------------------------------------------
_KEYWORD_MAP: dict[SkillName, list[str]] = {
    SkillName.FINANCEIRO: [
        # Faturamento
        r"\bfaturamento\b", r"\bfaturou\b", r"\bfaturar\b",
        # Receita
        r"\breceita\b", r"\breceit[ao]s\b",
        # Financeiro geral
        r"\bfinanceiro\b", r"\bfinanceira\b",
        # Custos e margem
        r"\bcusto[s]?\b", r"\bmargem\b", r"\bmargens\b", r"\bebitda\b",
        # Lucro
        r"\blucro\b", r"\blucr[ao]s\b", r"\brentabilidade\b",
        # Caixa
        r"\bcaixa\b", r"\bfluxo de caixa\b",
        # Resultado
        r"\bresultado financeiro\b", r"\bdem[oa]s financeiras\b",
        # MRR/ARR (SaaS)
        r"\bmrr\b", r"\barr\b", r"\brecurring revenue\b",
    ],
    SkillName.VENDAS: [
        # Vendas
        r"\bvendas?\b", r"\bvender\b", r"\bvendeu\b",
        # Ticket e pedidos
        r"\bticket\b", r"\bpedido[s]?\b", r"\bpediu\b",
        # Conversão e funil
        r"\bconvers[aã]o\b", r"\bfunil\b", r"\bpipeline\b", r"\bleads?\b",
        # Produtos
        r"\bprodutos? mais vendidos?\b", r"\btop produtos?\b",
        # Clientes novos
        r"\bclientes? novos?\b", r"\bnovos? clientes?\b",
        # Meta de vendas
        r"\bmeta[s]?\b", r"\btarget\b", r"\bquota\b",
    ],
    SkillName.CHURN: [
        # Cancelamento
        r"\bchurn\b", r"\bcancelamentos?\b", r"\bcancelou\b", r"\bcancelaram\b",
        # Retenção
        r"\breten[cç][aã]o\b", r"\bretidos?\b",
        # LTV / lifetime value
        r"\bltv\b", r"\blifetime value\b",
        # Risco e saída
        r"\bclientes? em risco\b", r"\bclientes? perdidos?\b",
        # Inadimplência
        r"\binadimpl[eê]ncia\b", r"\binadimplentes?\b",
    ],
}

# Prompt para fallback LLM quando keywords não são suficientes
_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Você é um roteador de perguntas executivas. Classifique a pergunta do CEO
em uma das categorias abaixo e responda APENAS com a palavra da categoria, sem pontuação:

- financeiro: faturamento, receita, custos, margem, lucro, fluxo de caixa, MRR
- vendas: pedidos, ticket médio, conversão, funil, leads, produtos mais vendidos
- churn: cancelamentos, clientes perdidos, retenção, LTV, inadimplência
- geral: perguntas amplas sobre o negócio que tocam múltiplas áreas
- desconhecido: não é possível identificar a intenção com os dados disponíveis

Responda APENAS com uma dessas palavras: financeiro, vendas, churn, geral, desconhecido"""
    ),
    ("human", "Pergunta do CEO: {query}"),
])


def route_skill(query: str) -> RouteResult:
    """
    Determina qual skill deve processar a pergunta do CEO.

    Fluxo:
    1. Tenta match por keywords (rápido, gratuito, determinístico)
    2. Se não encontrar, usa LLM para classificar (mais lento, mas flexível)
    3. Fallback para GERAL se LLM retornar valor inválido

    Args:
        query: Pergunta original do CEO em texto livre.

    Returns:
        RouteResult com a skill identificada e metadados de confiança.
    """
    query_lower = query.lower().strip()

    # --- Passo 1: Keyword matching ---
    keyword_result = _match_by_keywords(query_lower)
    if keyword_result:
        logger.info(
            "Skill identificada por keyword",
            skill=keyword_result.skill,
            query=query[:50],
        )
        return keyword_result

    # --- Passo 2: Fallback LLM ---
    logger.info(
        "Keyword não encontrada, usando LLM para roteamento",
        query=query[:50],
    )
    return _match_by_llm(query)


def _match_by_keywords(query_lower: str) -> RouteResult | None:
    """
    Tenta identificar a skill por correspondência de regex keywords.
    Retorna None se nenhuma keyword for encontrada.
    """
    scores: dict[SkillName, int] = {skill: 0 for skill in SkillName}

    for skill, patterns in _KEYWORD_MAP.items():
        for pattern in patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                scores[skill] += 1

    # Pega a skill com mais keywords encontradas
    best_skill = max(scores, key=lambda s: scores[s])
    best_score = scores[best_skill]

    if best_score == 0:
        return None

    # Se há empate entre skills, usa LLM para desempatar (retorna None aqui)
    top_skills = [s for s, score in scores.items() if score == best_score and score > 0]
    if len(top_skills) > 1:
        logger.debug(
            "Empate de keywords, escalando para LLM",
            tied_skills=[s.value for s in top_skills],
        )
        return None

    return RouteResult(
        skill=best_skill,
        confidence="high",
        method="keyword",
        original_query=query_lower,
    )


def _match_by_llm(query: str) -> RouteResult:
    """
    Usa o LLM para classificar a intenção quando keywords não são suficientes.
    Aplica retry simples e fallback para GERAL em caso de falha.
    """
    try:
        llm = get_llm()
        chain = _ROUTER_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"query": query}).strip().lower()

        # Valida o resultado contra o enum
        try:
            skill = SkillName(result)
        except ValueError:
            logger.warning(
                "LLM retornou valor inválido para roteamento",
                llm_output=result,
                fallback=SkillName.GERAL.value,
            )
            skill = SkillName.GERAL

        return RouteResult(
            skill=skill,
            confidence="medium",
            method="llm",
            original_query=query,
        )

    except Exception as exc:
        logger.error(
            "Falha no roteamento LLM, usando fallback GERAL",
            error=str(exc),
        )
        return RouteResult(
            skill=SkillName.GERAL,
            confidence="low",
            method="fallback",
            original_query=query,
        )
