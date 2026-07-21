# =============================================================================
# CEO Assistant — src/api/evolution.py
# Cliente HTTP para a Evolution API
#
# Por que encapsular em um cliente dedicado (e não usar requests direto)?
# - Centraliza a URL base, headers e autenticação em um único lugar
# - Facilita mocking nos testes (inject do cliente via DI)
# - Adiciona retry automático com backoff exponencial (tenacity)
# - Logging estruturado de todas as chamadas de saída
# - Fácil adicionar novos endpoints da Evolution API no futuro
# =============================================================================

import structlog
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import get_settings

logger = structlog.get_logger(__name__)


class EvolutionClient:
    """
    Cliente HTTP assíncrono para a Evolution API.

    Métodos disponíveis:
    - send_text_message(): envia texto para um número
    - get_instance_status(): verifica status da conexão Baileys
    - set_webhook(): configura webhook de uma instância (uso administrativo)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.evolution_base_url.rstrip("/")
        self._api_key = settings.evolution_api_key
        self._instance = settings.evolution_instance_name
        self._headers = {
            "apikey": self._api_key,
            "Content-Type": "application/json",
        }

    def _get_client(self) -> httpx.AsyncClient:
        """Cria cliente httpx com configuração padrão."""
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=httpx.Timeout(30.0, connect=5.0),
        )

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def send_text_message(
        self,
        to: str,
        text: str,
        delay_ms: int = 1200,
    ) -> dict:
        """
        Envia uma mensagem de texto via Evolution API.

        Args:
            to: JID do destinatário (ex: 5511999999999@s.whatsapp.net)
            text: Texto da mensagem (suporta formatação WhatsApp: *negrito*, _itálico_)
            delay_ms: Atraso em ms antes de enviar (simula digitação humana)

        Returns:
            Resposta da Evolution API (dict com id da mensagem, status, etc.)

        Raises:
            httpx.HTTPStatusError: Em caso de erro HTTP (4xx/5xx) após retries.
        """
        endpoint = f"/message/sendText/{self._instance}"
        payload = {
            "number": to,
            "text": text,
            "delay": delay_ms,
            "linkPreview": False,  # Evita preview de links para respostas limpas
        }

        logger.info(
            "Enviando mensagem WhatsApp",
            to=_mask_jid(to),
            text_preview=text[:50],
            endpoint=endpoint,
        )

        async with self._get_client() as client:
            response = await client.post(endpoint, json=payload)

        # Lança exceção em caso de erro HTTP
        response.raise_for_status()

        result = response.json()
        logger.info(
            "Mensagem enviada com sucesso",
            to=_mask_jid(to),
            message_id=result.get("key", {}).get("id", "unknown"),
        )
        return result

    async def get_instance_status(self) -> dict:
        """
        Verifica o status da conexão da instância Baileys.

        Returns:
            Dict com campo "state": "open" (conectado) | "close" | "connecting"
        """
        endpoint = f"/instance/connectionState/{self._instance}"

        async with self._get_client() as client:
            response = await client.get(endpoint)

        response.raise_for_status()
        result = response.json()

        state = result.get("instance", {}).get("state", "unknown")
        logger.info("Status da instância Evolution API", state=state, instance=self._instance)
        return result

    async def set_webhook(
        self,
        webhook_url: str,
        events: list[str] | None = None,
    ) -> dict:
        """
        Configura o webhook de uma instância.
        Útil para setup automatizado no startup da aplicação.

        Args:
            webhook_url: URL pública do endpoint de webhook (FastAPI)
            events: Lista de eventos para escutar. Default: ["MESSAGES_UPSERT"]

        Returns:
            Resposta da Evolution API.
        """
        if events is None:
            events = ["MESSAGES_UPSERT"]

        endpoint = f"/webhook/set/{self._instance}"
        payload = {
            "url": webhook_url,
            "webhook_by_events": False,
            "webhook_base64": False,
            "events": events,
        }

        async with self._get_client() as client:
            response = await client.post(endpoint, json=payload)

        response.raise_for_status()
        result = response.json()
        logger.info("Webhook configurado", url=webhook_url, events=events)
        return result

    async def send_typing_indicator(self, to: str) -> None:
        """
        Envia indicador de "digitando..." para melhor experiência do usuário.
        Fire-and-forget: falhas são silenciosas.
        """
        endpoint = f"/chat/presence/{self._instance}"
        payload = {
            "number": to,
            "options": {"presence": "composing", "delay": 3000},
        }

        try:
            async with self._get_client() as client:
                await client.post(endpoint, json=payload)
        except Exception as exc:
            # Não crítico — apenas log e continua
            logger.debug("Falha ao enviar indicador de digitação", error=str(exc))


# =============================================================================
# SINGLETON E FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

_client: EvolutionClient | None = None


def get_evolution_client() -> EvolutionClient:
    """Retorna instância singleton do EvolutionClient."""
    global _client
    if _client is None:
        _client = EvolutionClient()
    return _client


async def send_text_message(to: str, text: str) -> dict:
    """
    Função de conveniência para enviar texto.
    Usada pelos nós do agente LangGraph para desacoplar da classe.
    """
    client = get_evolution_client()
    return await client.send_text_message(to=to, text=text)


async def check_connection() -> bool:
    """
    Verifica se a instância WhatsApp está conectada.
    Retorna True se state == "open".
    """
    try:
        client = get_evolution_client()
        result = await client.get_instance_status()
        state = result.get("instance", {}).get("state", "unknown")
        return state == "open"
    except Exception as exc:
        logger.error("Falha ao verificar conexão Evolution API", error=str(exc))
        return False


def _mask_jid(jid: str) -> str:
    """Mascara JID para logging seguro."""
    if "@" not in jid:
        return "***"
    number, domain = jid.split("@", 1)
    if len(number) > 6:
        masked = number[:6] + "****" + number[-3:]
    else:
        masked = "***"
    return f"{masked}@{domain}"
