# =============================================================================
# CEO Assistant — src/skills/skill_financeiro.py
# Skill de Dados Financeiros
#
# Esta skill é responsável por:
# - Interpretar o período solicitado (mês, trimestre, ano, YTD)
# - Buscar dados financeiros (fictícios nesta versão de demonstração)
# - Retornar dados estruturados para o nó format_response do agente
#
# Por que dados fictícios aqui e não só no teste?
# Em produção, o método _fetch_data() seria substituído pela chamada ao
# conector real (Google Sheets, PostgreSQL, REST API de BI). Manter a
# interface igual permite trocar a fonte sem alterar a skill ou o agente.
#
# Por que a skill NÃO formata a resposta final?
# Responsabilidade única: a skill conhece DADOS, o nó format_response
# conhece APRESENTAÇÃO. Isso permite reutilizar a mesma skill com
# diferentes formatos de saída (WhatsApp, email, dashboard web).
# =============================================================================

import re
import arrow
import structlog
from dataclasses import dataclass, field
from typing import Any

logger = structlog.get_logger(__name__)


# =============================================================================
# DADOS FICTÍCIOS REALISTAS
# Simulam o dashboard financeiro de uma empresa SaaS B2B brasileira
# com ~R$ 30M ARR, crescendo ~10% a.a.
# =============================================================================

# Faturamento mensal (R$) — últimos 18 meses
_FATURAMENTO_MENSAL: dict[str, float] = {
    "2024-01": 2_210_450.00,
    "2024-02": 2_185_320.00,
    "2024-03": 2_290_870.00,
    "2024-04": 2_315_640.00,
    "2024-05": 2_380_120.00,
    "2024-06": 2_420_900.00,
    "2024-07": 2_510_340.00,
    "2024-08": 2_490_780.00,
    "2024-09": 2_560_430.00,
    "2024-10": 2_620_190.00,
    "2024-11": 2_580_340.00,
    "2024-12": 2_710_560.00,
    "2025-01": 2_530_180.00,
    "2025-02": 2_610_150.00,
    "2025-03": 2_680_340.00,
    "2025-04": 2_720_890.00,
    "2025-05": 2_780_450.00,
    "2025-06": 2_610_150.00,
    "2025-07": 2_847_320.00,
}

# Custos operacionais mensais (R$)
_CUSTOS_MENSAIS: dict[str, float] = {
    "2024-01": 1_547_315.00,
    "2024-02": 1_529_724.00,
    "2024-03": 1_603_609.00,
    "2024-04": 1_620_948.00,
    "2024-05": 1_666_084.00,
    "2024-06": 1_694_630.00,
    "2024-07": 1_757_238.00,
    "2024-08": 1_743_546.00,
    "2024-09": 1_792_301.00,
    "2024-10": 1_834_133.00,
    "2024-11": 1_806_238.00,
    "2024-12": 1_897_392.00,
    "2025-01": 1_771_126.00,
    "2025-02": 1_827_105.00,
    "2025-03": 1_876_238.00,
    "2025-04": 1_904_623.00,
    "2025-05": 1_946_315.00,
    "2025-06": 1_827_105.00,
    "2025-07": 1_993_124.00,
}

# Mapeamento de nomes de meses em português para número do mês
_MES_NOME_PARA_NUM: dict[str, int] = {
    "janeiro": 1, "jan": 1,
    "fevereiro": 2, "fev": 2,
    "março": 3, "mar": 3, "marco": 3,
    "abril": 4, "abr": 4,
    "maio": 5, "mai": 5,
    "junho": 6, "jun": 6,
    "julho": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "setembro": 9, "set": 9,
    "outubro": 10, "out": 10,
    "novembro": 11, "nov": 11,
    "dezembro": 12, "dez": 12,
}

# Mapeamento de trimestres
_TRIMESTRE_MESES: dict[str, list[int]] = {
    "q1": [1, 2, 3],
    "t1": [1, 2, 3],
    "primeiro trimestre": [1, 2, 3],
    "q2": [4, 5, 6],
    "t2": [4, 5, 6],
    "segundo trimestre": [4, 5, 6],
    "q3": [7, 8, 9],
    "t3": [7, 8, 9],
    "terceiro trimestre": [7, 8, 9],
    "q4": [10, 11, 12],
    "t4": [10, 11, 12],
    "quarto trimestre": [10, 11, 12],
}


@dataclass
class PeriodoFinanceiro:
    """Representa um período temporal resolvido para chaves do dict de dados."""
    chaves: list[str]          # lista de "YYYY-MM" abrangidas pelo período
    descricao: str             # descrição humana (ex: "julho/2025")
    tipo: str                  # "mensal" | "trimestral" | "anual" | "ytd"


class SkillFinanceiro:
    """
    Skill de consulta de dados financeiros.

    Responsabilidades:
    - Extrair o período da intenção (ou da query diretamente)
    - Buscar dados financeiros para o período
    - Calcular métricas derivadas (margem, variação)
    - Retornar dict estruturado para o nó format_response

    Uso:
        skill = SkillFinanceiro()
        data = skill.execute(query="qual foi o faturamento de julho?", intent={...})
    """

    def execute(self, query: str, intent: dict[str, Any]) -> dict[str, Any]:
        """
        Executa a consulta financeira.

        Args:
            query: Pergunta original do CEO.
            intent: Dicionário de intenção extraído pelo nó interpret_intent.

        Returns:
            Dicionário com dados financeiros estruturados e metadados de fonte.
        """
        logger.info("Executando skill financeira", query=query[:50])

        # 1. Resolve o período solicitado
        periodo = self._resolve_periodo(query=query, intent=intent)

        if not periodo.chaves:
            return {
                "erro": "Não encontrei dados para o período solicitado.",
                "periodo_solicitado": intent.get("periodo", "não identificado"),
                "periodos_disponiveis": self._periodos_disponiveis(),
                "source": "Dashboard Financeiro (dados de demonstração)",
            }

        # 2. Busca dados para o período
        dados = self._fetch_data(periodo)

        # 3. Calcula período anterior para comparação (quando aplicável)
        dados_anteriores = self._fetch_periodo_anterior(periodo)

        # 4. Monta resposta estruturada
        result = self._build_result(
            periodo=periodo,
            dados=dados,
            dados_anteriores=dados_anteriores,
            query=query,
        )

        logger.info(
            "Skill financeira concluída",
            periodo=periodo.descricao,
            faturamento=dados.get("faturamento_total"),
        )
        return result

    # -------------------------------------------------------------------------
    # Resolução de período
    # -------------------------------------------------------------------------

    def _resolve_periodo(self, query: str, intent: dict[str, Any]) -> PeriodoFinanceiro:
        """
        Resolve o período mencionado na query ou intenção para chaves do dict de dados.

        Estratégia:
        1. Tenta usar o período extraído pelo LLM (intent["periodo"])
        2. Se falhar, tenta extrair da query com regex
        3. Se ainda falhar, usa o mês corrente
        """
        query_lower = query.lower()
        periodo_hint = (intent.get("periodo") or "").lower()

        # Tenta trimestre primeiro (Q1, Q2, etc.)
        trimestre = self._parse_trimestre(query_lower + " " + periodo_hint)
        if trimestre:
            return trimestre

        # Tenta mês específico
        mes = self._parse_mes(query_lower + " " + periodo_hint)
        if mes:
            return mes

        # Tenta ano
        ano = self._parse_ano(query_lower + " " + periodo_hint)
        if ano:
            return ano

        # Tenta YTD ("ano até hoje", "YTD", "esse ano")
        ytd = self._parse_ytd(query_lower + " " + periodo_hint)
        if ytd:
            return ytd

        # Fallback: mês mais recente disponível
        ultima_chave = sorted(_FATURAMENTO_MENSAL.keys())[-1]
        ano_u, mes_u = ultima_chave.split("-")
        return PeriodoFinanceiro(
            chaves=[ultima_chave],
            descricao=f"{_num_para_mes(int(mes_u))}/{ano_u}",
            tipo="mensal",
        )

    def _parse_mes(self, texto: str) -> PeriodoFinanceiro | None:
        """Extrai mês e ano de um texto livre."""
        agora = arrow.now()

        # Padrão: "julho de 2025", "julho/2025", "07/2025", "julho 2025"
        for nome_mes, num_mes in _MES_NOME_PARA_NUM.items():
            # Nome do mês + ano (ex: "julho de 2025", "julho/2025", "julho 2025")
            pattern = rf"{nome_mes}[\s/de]*(\d{{4}})?"
            match = re.search(pattern, texto, re.IGNORECASE)
            if match:
                ano = int(match.group(1)) if match.group(1) else agora.year
                chave = f"{ano:04d}-{num_mes:02d}"
                if chave in _FATURAMENTO_MENSAL:
                    return PeriodoFinanceiro(
                        chaves=[chave],
                        descricao=f"{nome_mes.capitalize()}/{ano}",
                        tipo="mensal",
                    )

        # Padrão numérico: "07/2025", "7/2025"
        match = re.search(r"\b(\d{1,2})/(\d{4})\b", texto)
        if match:
            num_mes, ano = int(match.group(1)), int(match.group(2))
            if 1 <= num_mes <= 12:
                chave = f"{ano:04d}-{num_mes:02d}"
                if chave in _FATURAMENTO_MENSAL:
                    return PeriodoFinanceiro(
                        chaves=[chave],
                        descricao=f"{_num_para_mes(num_mes)}/{ano}",
                        tipo="mensal",
                    )

        return None

    def _parse_trimestre(self, texto: str) -> PeriodoFinanceiro | None:
        """Extrai trimestre de um texto livre."""
        agora = arrow.now()

        for nome_trim, meses in _TRIMESTRE_MESES.items():
            if nome_trim in texto:
                # Tenta extrair o ano
                match = re.search(r"\d{4}", texto)
                ano = int(match.group()) if match else agora.year

                chaves = [f"{ano:04d}-{m:02d}" for m in meses]
                chaves_validas = [c for c in chaves if c in _FATURAMENTO_MENSAL]

                if chaves_validas:
                    return PeriodoFinanceiro(
                        chaves=chaves_validas,
                        descricao=f"{nome_trim.upper()} {ano}",
                        tipo="trimestral",
                    )
        return None

    def _parse_ano(self, texto: str) -> PeriodoFinanceiro | None:
        """Extrai um ano completo do texto."""
        match = re.search(r"\b(202[0-9])\b", texto)
        if not match:
            return None

        ano = int(match.group(1))
        chaves = [f"{ano:04d}-{m:02d}" for m in range(1, 13)]
        chaves_validas = [c for c in chaves if c in _FATURAMENTO_MENSAL]

        if chaves_validas:
            return PeriodoFinanceiro(
                chaves=chaves_validas,
                descricao=f"Ano {ano}",
                tipo="anual",
            )
        return None

    def _parse_ytd(self, texto: str) -> PeriodoFinanceiro | None:
        """Extrai período YTD (year-to-date) do texto."""
        ytd_keywords = ["ytd", "ano até hoje", "esse ano", "este ano", "ano corrente"]
        if not any(k in texto for k in ytd_keywords):
            return None

        agora = arrow.now()
        chaves = [
            f"{agora.year:04d}-{m:02d}"
            for m in range(1, agora.month + 1)
        ]
        chaves_validas = [c for c in chaves if c in _FATURAMENTO_MENSAL]

        if chaves_validas:
            return PeriodoFinanceiro(
                chaves=chaves_validas,
                descricao=f"YTD {agora.year} (Jan–{_num_para_mes(agora.month)})",
                tipo="ytd",
            )
        return None

    # -------------------------------------------------------------------------
    # Busca de dados
    # -------------------------------------------------------------------------

    def _fetch_data(self, periodo: PeriodoFinanceiro) -> dict[str, Any]:
        """
        Busca dados financeiros para o período resolvido.

        Em produção: substituir os dicts locais por chamadas ao
        conector real (Google Sheets, PostgreSQL, API REST).
        """
        faturamento_total = sum(
            _FATURAMENTO_MENSAL.get(k, 0) for k in periodo.chaves
        )
        custos_total = sum(
            _CUSTOS_MENSAIS.get(k, 0) for k in periodo.chaves
        )
        lucro_bruto = faturamento_total - custos_total
        margem = (lucro_bruto / faturamento_total * 100) if faturamento_total > 0 else 0

        return {
            "faturamento_total": faturamento_total,
            "custos_total": custos_total,
            "lucro_bruto": lucro_bruto,
            "margem_percentual": round(margem, 1),
            "num_meses": len(periodo.chaves),
        }

    def _fetch_periodo_anterior(self, periodo: PeriodoFinanceiro) -> dict[str, Any] | None:
        """
        Calcula o período anterior equivalente para comparação.
        Ex: julho/2025 → junho/2025 | Q3 2025 → Q2 2025
        """
        if periodo.tipo == "mensal" and len(periodo.chaves) == 1:
            data = arrow.get(periodo.chaves[0], "YYYY-MM").shift(months=-1)
            chave_ant = data.format("YYYY-MM")
            if chave_ant in _FATURAMENTO_MENSAL:
                return {
                    "faturamento_total": _FATURAMENTO_MENSAL[chave_ant],
                    "custos_total": _CUSTOS_MENSAIS.get(chave_ant, 0),
                    "chave": chave_ant,
                }
        return None

    # -------------------------------------------------------------------------
    # Construção da resposta
    # -------------------------------------------------------------------------

    def _build_result(
        self,
        periodo: PeriodoFinanceiro,
        dados: dict[str, Any],
        dados_anteriores: dict[str, Any] | None,
        query: str,
    ) -> dict[str, Any]:
        """Monta o dicionário final com todos os dados e metadados."""
        result: dict[str, Any] = {
            "periodo": periodo.descricao,
            "tipo_periodo": periodo.tipo,
            "faturamento": _fmt_brl(dados["faturamento_total"]),
            "custos": _fmt_brl(dados["custos_total"]),
            "lucro_bruto": _fmt_brl(dados["lucro_bruto"]),
            "margem_bruta": f"{dados['margem_percentual']}%",
            "source": "Dashboard Financeiro (dados de demonstração)",
            "ultima_atualizacao": arrow.now("America/Sao_Paulo").format("DD/MM/YYYY [às] HH[h]"),
        }

        # Adiciona comparação com período anterior (se disponível)
        if dados_anteriores:
            fat_ant = dados_anteriores["faturamento_total"]
            fat_atual = dados["faturamento_total"]
            variacao = ((fat_atual - fat_ant) / fat_ant * 100) if fat_ant > 0 else 0
            sinal = "+" if variacao >= 0 else ""
            emoji = "🟢" if variacao >= 0 else "🔴"

            result["comparacao_periodo_anterior"] = {
                "faturamento_anterior": _fmt_brl(fat_ant),
                "variacao_percentual": f"{sinal}{variacao:.1f}%",
                "tendencia": emoji,
            }

        return result

    def _periodos_disponiveis(self) -> str:
        """Retorna string descritiva dos períodos disponíveis nos dados."""
        chaves = sorted(_FATURAMENTO_MENSAL.keys())
        if not chaves:
            return "nenhum"
        return f"{chaves[0]} a {chaves[-1]}"


# =============================================================================
# HELPERS
# =============================================================================

def _fmt_brl(valor: float) -> str:
    """
    Formata um float como moeda brasileira: R$ 2.847.320,00
    Estratégia: formata com separador de milhar americano (,) e
    então converte para o padrão brasileiro (. para milhar, , para decimal).
    """
    # Formata: 2847320.00 → "2,847,320.00"
    american = f"{valor:,.2f}"
    # Converte: "2,847,320.00" → "2.847.320,00"
    return "R$ " + american.replace(",", "X").replace(".", ",").replace("X", ".")


def _num_para_mes(num: int) -> str:
    """Converte número do mês para nome em português."""
    nomes = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
        5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
        9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
    }
    return nomes.get(num, str(num))
