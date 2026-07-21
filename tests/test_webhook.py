# =============================================================================
# CEO Assistant — tests/test_webhook.py
# Testes do Endpoint de Webhook e Segurança
# =============================================================================

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.main import app


# Cliente de teste síncrono do FastAPI
@pytest.fixture
def client():
    return TestClient(app)


# =============================================================================
# PAYLOADS DE EXEMPLO
# =============================================================================

def make_payload(
    remote_jid: str = "5511999999999@s.whatsapp.net",
    text: str = "Qual o faturamento de julho?",
    from_me: bool = False,
    event: str = "messages.upsert",
    message_type: str = "conversation",
) -> dict:
    """Fábrica de payloads de webhook para testes."""
    return {
        "event": event,
        "instance": "WHATSAPP-BAILEYS",
        "data": {
            "key": {
                "remoteJid": remote_jid,
                "fromMe": from_me,
                "id": "TESTMSGID001",
            },
            "message": {
                "conversation": text if message_type == "conversation" else None,
            },
            "messageType": message_type,
            "messageTimestamp": 1721563200,
        },
    }


# =============================================================================
# TESTES: PARSING DO PAYLOAD
# =============================================================================

class TestPayloadParsing:

    def test_parse_valid_text_message(self, client):
        """Deve aceitar e parsear payload de mensagem de texto válido."""
        with patch("src.api.webhook._process_message_background", new_callable=AsyncMock):
            with patch("src.api.security.SecurityMiddleware.is_authorized", return_value=True):
                with patch("src.api.security.SecurityMiddleware.is_rate_limited", return_value=False):
                    response = client.post("/webhook/evolution", json=make_payload())
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_ignore_non_message_event(self, client):
        """Deve ignorar eventos que não são messages.upsert."""
        payload = make_payload(event="connection.update")
        response = client.post("/webhook/evolution", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_ignore_message_from_me(self, client):
        """Deve ignorar mensagens enviadas pelo próprio bot (fromMe=True)."""
        payload = make_payload(from_me=True)
        response = client.post("/webhook/evolution", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_ignore_group_message(self, client):
        """Deve ignorar mensagens de grupos (@g.us)."""
        payload = make_payload(remote_jid="5511999999999-1234567890@g.us")
        response = client.post("/webhook/evolution", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_ignore_audio_message(self, client):
        """Deve ignorar mensagens de áudio (não suportadas ainda)."""
        payload = make_payload(message_type="audioMessage")
        payload["data"]["message"] = {
            "audioMessage": {
                "url": "https://example.com/audio.ogg",
                "mimetype": "audio/ogg; codecs=opus",
                "ptt": True,
                "seconds": 10,
            }
        }
        with patch("src.api.security.SecurityMiddleware.is_authorized", return_value=True):
            with patch("src.api.security.SecurityMiddleware.is_rate_limited", return_value=False):
                response = client.post("/webhook/evolution", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_ignore_sticker_message(self, client):
        """Deve ignorar figurinhas."""
        payload = make_payload(message_type="stickerMessage")
        payload["data"]["message"] = {"stickerMessage": {"url": "...", "mimetype": "image/webp"}}
        with patch("src.api.security.SecurityMiddleware.is_authorized", return_value=True):
            with patch("src.api.security.SecurityMiddleware.is_rate_limited", return_value=False):
                response = client.post("/webhook/evolution", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_invalid_json_returns_422(self, client):
        """Payload com JSON inválido deve retornar 422."""
        response = client.post(
            "/webhook/evolution",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# =============================================================================
# TESTES: SEGURANÇA — WHITELIST
# =============================================================================

class TestSecurityWhitelist:

    def test_reject_unauthorized_number(self, client):
        """Deve rejeitar número não autorizado."""
        payload = make_payload(remote_jid="5599888888888@s.whatsapp.net")
        # NÃO mockamos is_authorized para testar a validação real
        with patch(
            "src.config.Settings.ceo_whatsapp_number",
            new_callable=lambda: property(lambda self: "5511999999999@s.whatsapp.net"),
        ):
            response = client.post("/webhook/evolution", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_accept_authorized_number(self, client):
        """Deve aceitar o número autorizado do CEO."""
        with patch("src.api.webhook._process_message_background", new_callable=AsyncMock):
            with patch("src.api.security.SecurityMiddleware.is_authorized", return_value=True):
                with patch("src.api.security.SecurityMiddleware.is_rate_limited", return_value=False):
                    response = client.post(
                        "/webhook/evolution",
                        json=make_payload(remote_jid="5511999999999@s.whatsapp.net"),
                    )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_reject_rate_limited_number(self, client):
        """Deve rejeitar número que excedeu o rate limit."""
        with patch("src.api.security.SecurityMiddleware.is_authorized", return_value=True):
            with patch("src.api.security.SecurityMiddleware.is_rate_limited", return_value=True):
                response = client.post("/webhook/evolution", json=make_payload())
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"


# =============================================================================
# TESTES: MODELS — extract_text
# =============================================================================

class TestMessageContentExtractText:

    def test_extract_from_conversation(self):
        """Deve extrair texto do campo conversation."""
        from src.api.models import MessageContent
        content = MessageContent(conversation="Qual o faturamento de julho?")
        assert content.extract_text() == "Qual o faturamento de julho?"

    def test_extract_from_extended_text(self):
        """Deve extrair texto de extendedTextMessage."""
        from src.api.models import MessageContent, ExtendedTextMessage
        content = MessageContent(
            extendedTextMessage=ExtendedTextMessage(text="Como estamos no Q3?")
        )
        assert content.extract_text() == "Como estamos no Q3?"

    def test_extract_from_image_caption(self):
        """Deve extrair legenda de imagem como texto."""
        from src.api.models import MessageContent, ImageMessage
        content = MessageContent(
            imageMessage=ImageMessage(url="...", caption="Analise este gráfico")
        )
        assert content.extract_text() == "Analise este gráfico"

    def test_returns_none_for_audio(self):
        """Deve retornar None para mensagens de áudio (sem texto)."""
        from src.api.models import MessageContent, AudioMessage
        content = MessageContent(audioMessage=AudioMessage(ptt=True, seconds=5))
        assert content.extract_text() is None

    def test_get_message_type_text(self):
        """Deve identificar tipo text corretamente."""
        from src.api.models import MessageContent
        content = MessageContent(conversation="oi")
        assert content.get_message_type() == "text"

    def test_get_message_type_audio(self):
        """Deve identificar tipo audio corretamente."""
        from src.api.models import MessageContent, AudioMessage
        content = MessageContent(audioMessage=AudioMessage(ptt=True))
        assert content.get_message_type() == "audio"


# =============================================================================
# TESTES: ENDPOINTS UTILITÁRIOS
# =============================================================================

class TestUtilityEndpoints:

    def test_healthcheck(self, client):
        """GET /health deve retornar 200 com status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
