# =============================================================================
# CEO Assistant — tests/test_skill_financeiro.py
# Testes da Skill Financeira
# =============================================================================

import pytest
from src.skills.skill_financeiro import SkillFinanceiro, _fmt_brl, _num_para_mes


@pytest.fixture
def skill():
    return SkillFinanceiro()


# =============================================================================
# TESTES: RESOLUÇÃO DE PERÍODO
# =============================================================================

class TestPeriodoResolution:

    def test_resolve_month_by_name(self, skill):
        """Deve resolver 'julho' para julho do ano corrente."""
        result = skill.execute(
            query="Qual o faturamento de julho?",
            intent={"periodo": "julho/2025", "entidade": "faturamento", "tipo": "pontual", "filtros": None},
        )
        assert "periodo" in result
        assert "julho" in result["periodo"].lower() or "jul" in result["periodo"].lower()

    def test_resolve_month_july_2025(self, skill):
        """Deve retornar faturamento correto de julho/2025."""
        result = skill.execute(
            query="faturamento de julho de 2025",
            intent={"periodo": "julho/2025"},
        )
        assert result.get("faturamento") == "R$ 2.847.320,00"

    def test_resolve_month_june_2025(self, skill):
        """Deve retornar faturamento correto de junho/2025."""
        result = skill.execute(
            query="faturamento de junho de 2025",
            intent={"periodo": "junho/2025"},
        )
        assert result.get("faturamento") == "R$ 2.610.150,00"

    def test_resolve_quarter_q2_2025(self, skill):
        """Deve somar corretamente os 3 meses do Q2/2025."""
        result = skill.execute(
            query="faturamento do Q2 2025",
            intent={"periodo": "Q2 2025"},
        )
        # Q2 2025 = Abril + Maio + Junho = 2.720.890 + 2.780.450 + 2.610.150
        expected_total = 2_720_890.00 + 2_780_450.00 + 2_610_150.00
        assert result.get("tipo_periodo") == "trimestral"
        # Verifica valor numérico extraindo do string formatado
        fat_str = result.get("faturamento", "").replace("R$ ", "").replace(".", "").replace(",", ".")
        assert abs(float(fat_str) - expected_total) < 1.0

    def test_resolve_full_year_2024(self, skill):
        """Deve somar todos os meses de 2024."""
        result = skill.execute(
            query="qual foi o faturamento em 2024?",
            intent={"periodo": "2024"},
        )
        assert result.get("tipo_periodo") == "anual"
        assert result.get("periodo") == "Ano 2024"

    def test_unknown_period_returns_error(self, skill):
        """Período sem dados deve retornar informação de erro."""
        result = skill.execute(
            query="faturamento de janeiro de 2020",
            intent={"periodo": "janeiro/2020"},
        )
        # 2020 não está nos dados fictícios
        assert "erro" in result or "periodos_disponiveis" in result

    def test_comparison_with_previous_month(self, skill):
        """Consulta mensal deve incluir comparação com mês anterior."""
        result = skill.execute(
            query="faturamento de julho 2025",
            intent={"periodo": "julho/2025"},
        )
        assert "comparacao_periodo_anterior" in result
        comp = result["comparacao_periodo_anterior"]
        assert "faturamento_anterior" in comp
        assert "variacao_percentual" in comp
        assert "tendencia" in comp

    def test_july_growth_vs_june(self, skill):
        """Julho/2025 deve mostrar crescimento vs junho/2025."""
        result = skill.execute(
            query="faturamento de julho 2025",
            intent={"periodo": "julho/2025"},
        )
        comp = result.get("comparacao_periodo_anterior", {})
        variacao = comp.get("variacao_percentual", "")
        # Julho 2.847.320 > Junho 2.610.150 = crescimento positivo
        assert variacao.startswith("+")
        assert comp.get("tendencia") == "🟢"


# =============================================================================
# TESTES: CÁLCULO DE MÉTRICAS
# =============================================================================

class TestMetricsCalculation:

    def test_margin_is_calculated(self, skill):
        """Resultado deve incluir margem bruta calculada."""
        result = skill.execute(
            query="faturamento julho",
            intent={"periodo": "julho/2025"},
        )
        assert "margem_bruta" in result
        margin_str = result["margem_bruta"].replace("%", "")
        margin = float(margin_str)
        assert 0 < margin < 100, "Margem deve estar entre 0% e 100%"

    def test_profit_equals_revenue_minus_costs(self, skill):
        """Lucro bruto deve ser faturamento - custos."""
        result = skill.execute(
            query="resultados de julho 2025",
            intent={"periodo": "julho/2025"},
        )

        def parse_brl(s: str) -> float:
            return float(s.replace("R$ ", "").replace(".", "").replace(",", "."))

        fat = parse_brl(result["faturamento"])
        custo = parse_brl(result["custos"])
        lucro = parse_brl(result["lucro_bruto"])
        assert abs(lucro - (fat - custo)) < 1.0

    def test_result_has_source_field(self, skill):
        """Resultado deve sempre incluir o campo 'source'."""
        result = skill.execute(
            query="faturamento julho",
            intent={"periodo": "julho/2025"},
        )
        assert "source" in result
        assert len(result["source"]) > 0

    def test_result_has_update_timestamp(self, skill):
        """Resultado deve incluir timestamp de última atualização."""
        result = skill.execute(
            query="faturamento julho",
            intent={"periodo": "julho/2025"},
        )
        assert "ultima_atualizacao" in result


# =============================================================================
# TESTES: HELPERS
# =============================================================================

class TestHelpers:

    @pytest.mark.parametrize("valor,expected", [
        (2_847_320.00, "R$ 2.847.320,00"),
        (1_000_000.00, "R$ 1.000.000,00"),
        (500.50, "R$ 500,50"),
        (0.0, "R$ 0,00"),
    ])
    def test_fmt_brl(self, valor: float, expected: str):
        """Formatação de moeda brasileira deve estar correta."""
        assert _fmt_brl(valor) == expected

    @pytest.mark.parametrize("num,expected", [
        (1, "Janeiro"), (7, "Julho"), (12, "Dezembro"),
    ])
    def test_num_para_mes(self, num: int, expected: str):
        """Conversão de número para nome do mês deve funcionar."""
        assert _num_para_mes(num) == expected
