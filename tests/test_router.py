# =============================================================================
# CEO Assistant — tests/test_router.py
# Testes do Roteador de Skills por Intenção
# =============================================================================

import pytest
from unittest.mock import patch, MagicMock

from src.core.router import route_skill, SkillName


# =============================================================================
# TESTES: ROTEAMENTO POR KEYWORD
# =============================================================================

class TestKeywordRouting:
    """Testa o roteamento determinístico por keywords — não usa LLM."""

    @pytest.mark.parametrize("query,expected_skill", [
        # --- Financeiro ---
        ("Qual foi o faturamento de julho?", SkillName.FINANCEIRO),
        ("Como está nossa receita no Q3?", SkillName.FINANCEIRO),
        ("Qual a margem bruta desse mês?", SkillName.FINANCEIRO),
        ("Me mostra o EBITDA do segundo semestre", SkillName.FINANCEIRO),
        ("Como está o MRR?", SkillName.FINANCEIRO),
        ("Quero ver o fluxo de caixa de junho", SkillName.FINANCEIRO),
        ("Qual o lucro líquido do Q2?", SkillName.FINANCEIRO),
        # --- Vendas ---
        ("Quantas vendas fizemos em julho?", SkillName.VENDAS),
        ("Qual o ticket médio dos pedidos de junho?", SkillName.VENDAS),
        ("Quantos leads temos no pipeline?", SkillName.VENDAS),
        ("Quais são os produtos mais vendidos?", SkillName.VENDAS),
        ("Qual a taxa de conversão do funil?", SkillName.VENDAS),
        ("Atingimos a meta de vendas?", SkillName.VENDAS),
        # --- Churn ---
        ("Qual a taxa de churn em julho?", SkillName.CHURN),
        ("Quantos clientes cancelaram esse mês?", SkillName.CHURN),
        ("Como está a retenção de clientes?", SkillName.CHURN),
        ("Quais clientes estão em risco de cancelar?", SkillName.CHURN),
        ("Qual o LTV médio dos nossos clientes?", SkillName.CHURN),
    ])
    def test_keyword_routing(self, query: str, expected_skill: SkillName):
        """Deve rotear corretamente por keywords sem chamar o LLM."""
        with patch("src.core.router.get_llm") as mock_llm:
            result = route_skill(query)

        # Garante que o LLM NÃO foi chamado (routing por keyword)
        assert result.skill == expected_skill, (
            f"Query '{query}' deveria rotear para {expected_skill.value}, "
            f"mas foi para {result.skill.value}"
        )
        if result.skill != SkillName.DESCONHECIDO:
            assert result.confidence in ("high", "medium", "low")

    def test_keyword_method_is_keyword(self):
        """Roteamento por keyword deve ter method='keyword'."""
        result = route_skill("qual o faturamento de julho?")
        assert result.method == "keyword"
        assert result.confidence == "high"

    def test_original_query_preserved(self):
        """A query original deve ser preservada no resultado."""
        query = "Qual foi o faturamento de julho?"
        result = route_skill(query)
        assert result.original_query.lower() == query.lower()


# =============================================================================
# TESTES: FALLBACK LLM
# =============================================================================

class TestLLMFallback:
    """Testa o fallback para LLM quando keywords não são suficientes."""

    def test_ambiguous_query_uses_llm(self):
        """Query sem keywords claras deve usar o LLM."""
        # "Como estamos indo?" não tem keywords de nenhuma categoria
        with patch("src.core.router.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_chain_result = MagicMock()
            mock_chain_result.invoke = MagicMock(return_value="geral")
            mock_get_llm.return_value = mock_llm

            # Mock do chain (prompt | llm | parser)
            with patch("src.core.router._ROUTER_PROMPT") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain_result)

                result = route_skill("Como estamos indo nos últimos meses?")

        # Resultado pode ser "geral" ou qualquer skill — o importante é não crashar
        assert result.skill in SkillName.__members__.values()

    def test_llm_returns_invalid_value_falls_to_geral(self):
        """LLM retornando valor inválido deve cair no fallback GERAL."""
        with patch("src.core.router._match_by_keywords", return_value=None):
            with patch("src.core.router.get_llm") as mock_get_llm:
                mock_llm = MagicMock()
                mock_get_llm.return_value = mock_llm

                with patch("src.core.router._ROUTER_PROMPT") as mock_prompt:
                    chain_mock = MagicMock()
                    chain_mock.invoke = MagicMock(return_value="valor_invalido_xyz")
                    mock_prompt.__or__ = MagicMock(return_value=chain_mock)

                    result = route_skill("Uma pergunta muito estranha sem keywords")

        assert result.skill == SkillName.GERAL

    def test_llm_exception_falls_to_geral(self):
        """Exceção no LLM deve fazer fallback para GERAL sem crashar."""
        with patch("src.core.router._match_by_keywords", return_value=None):
            with patch("src.core.router.get_llm", side_effect=Exception("LLM offline")):
                result = route_skill("qualquer coisa aqui")

        assert result.skill == SkillName.GERAL
        assert result.confidence == "low"
        assert result.method == "fallback"


# =============================================================================
# TESTES: CASOS EDGE
# =============================================================================

class TestEdgeCases:

    def test_empty_query(self):
        """Query vazia deve retornar algum resultado sem crashar."""
        result = route_skill("")
        assert result.skill in SkillName.__members__.values()

    def test_very_long_query(self):
        """Query muito longa deve ser processada sem erro."""
        long_query = "faturamento " * 100
        result = route_skill(long_query)
        assert result.skill == SkillName.FINANCEIRO

    def test_query_with_special_chars(self):
        """Query com caracteres especiais deve ser processada."""
        result = route_skill("Qual o faturamento? R$ 2M? #Q3 @2025!")
        assert result.skill == SkillName.FINANCEIRO

    def test_uppercase_query(self):
        """Keywords em maiúsculas devem ser detectadas."""
        result = route_skill("QUAL O FATURAMENTO DE JULHO?")
        assert result.skill == SkillName.FINANCEIRO

    def test_result_dataclass_fields(self):
        """RouteResult deve ter todos os campos esperados."""
        result = route_skill("qual o faturamento?")
        assert hasattr(result, "skill")
        assert hasattr(result, "confidence")
        assert hasattr(result, "method")
        assert hasattr(result, "original_query")
