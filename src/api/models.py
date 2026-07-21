# =============================================================================
# CEO Assistant — src/api/models.py
# Modelos Pydantic para parsing do payload da Evolution API
#
# Por que modelar o payload com Pydantic?
# A Evolution API pode enviar payloads com estruturas variadas dependendo
# do tipo de mensagem (texto, áudio, imagem, sticker, reação, etc.).
# O Pydantic garante:
# - Validação automática dos tipos e campos obrigatórios
# - Falha rápida (422) em caso de payload malformado
# - Documentação implícita da estrutura esperada
# - Type hints para autocomplete e análise estática
#
# IMPORTANTE: Trate o payload como podendo variar!
# Use Optional em todos os campos que não garantimos presença.
# =============================================================================

from typing import Any, Literal
from pydantic import BaseModel, Field


# =============================================================================
# SUBMODELOS — partes do payload messages.upsert
# =============================================================================

class MessageKey(BaseModel):
    """Identifica unicamente uma mensagem no WhatsApp."""
    remoteJid: str = Field(..., description="Número do remetente: 5511999@s.whatsapp.net")
    fromMe: bool = Field(False, description="True se a mensagem foi enviada pelo bot")
    id: str = Field(..., description="ID único da mensagem no WhatsApp")
    participant: str | None = Field(None, description="Em grupos: JID do participante")


class TextMessage(BaseModel):
    """Mensagem de texto simples (conversation ou extendedTextMessage)."""
    conversation: str | None = Field(None, description="Texto direto (mensagens simples)")


class ExtendedTextMessage(BaseModel):
    """Mensagem de texto com formatação ou citação."""
    text: str | None = Field(None, description="Texto da mensagem estendida")
    contextInfo: dict[str, Any] | None = Field(None, description="Contexto de resposta/citação")


class AudioMessage(BaseModel):
    """Mensagem de áudio (PTT = Push-To-Talk = mensagem de voz)."""
    url: str | None = Field(None, description="URL do áudio no servidor do WhatsApp")
    mimetype: str | None = Field(None, description="audio/ogg; codecs=opus")
    ptt: bool | None = Field(None, description="True = mensagem de voz gravada")
    seconds: int | None = Field(None, description="Duração em segundos")


class ImageMessage(BaseModel):
    """Mensagem com imagem."""
    url: str | None = Field(None, description="URL da imagem")
    mimetype: str | None = Field(None, description="image/jpeg, image/png, etc.")
    caption: str | None = Field(None, description="Legenda da imagem (opcional)")
    height: int | None = None
    width: int | None = None


class DocumentMessage(BaseModel):
    """Mensagem com documento (PDF, Excel, etc.)."""
    url: str | None = None
    mimetype: str | None = None
    title: str | None = None
    fileName: str | None = None


class StickerMessage(BaseModel):
    """Mensagem com sticker/figurinha."""
    url: str | None = None
    mimetype: str | None = None
    isAnimated: bool | None = None


class ReactionMessage(BaseModel):
    """Reação a uma mensagem."""
    text: str | None = Field(None, description="Emoji da reação")
    key: dict[str, Any] | None = None


class MessageContent(BaseModel):
    """
    Conteúdo da mensagem — agrupa todos os tipos possíveis.
    Apenas um desses campos estará preenchido por mensagem.
    """
    # Texto simples (campo mais comum)
    conversation: str | None = None

    # Texto estendido (com formatação, resposta, link preview)
    extendedTextMessage: ExtendedTextMessage | None = None

    # Áudio
    audioMessage: AudioMessage | None = None

    # Imagem
    imageMessage: ImageMessage | None = None

    # Documento
    documentMessage: DocumentMessage | None = None
    documentWithCaptionMessage: dict[str, Any] | None = None

    # Sticker
    stickerMessage: StickerMessage | None = None

    # Reação
    reactionMessage: ReactionMessage | None = None

    # Outros tipos (poll, location, contact, etc.) — capturados genericamente
    pollCreationMessage: dict[str, Any] | None = None
    locationMessage: dict[str, Any] | None = None
    contactMessage: dict[str, Any] | None = None

    def extract_text(self) -> str | None:
        """
        Extrai o texto da mensagem independentemente do tipo.

        Returns:
            Texto extraído ou None se a mensagem não for de texto.
        """
        # Prioridade: conversation > extendedTextMessage > legenda de imagem
        if self.conversation:
            return self.conversation.strip()

        if self.extendedTextMessage and self.extendedTextMessage.text:
            return self.extendedTextMessage.text.strip()

        if self.imageMessage and self.imageMessage.caption:
            return self.imageMessage.caption.strip()

        return None

    def get_message_type(self) -> str:
        """
        Retorna o tipo da mensagem para logging e roteamento.

        Returns:
            String descritiva: "text", "audio", "image", "document",
            "sticker", "reaction", "other"
        """
        if self.conversation or self.extendedTextMessage:
            return "text"
        if self.audioMessage:
            return "audio"
        if self.imageMessage:
            return "image"
        if self.documentMessage or self.documentWithCaptionMessage:
            return "document"
        if self.stickerMessage:
            return "sticker"
        if self.reactionMessage:
            return "reaction"
        return "other"


# =============================================================================
# MODELO PRINCIPAL — payload do webhook messages.upsert
# =============================================================================

class MessageData(BaseModel):
    """
    Dados de uma mensagem individual dentro do evento messages.upsert.
    """
    key: MessageKey
    message: MessageContent | None = Field(
        None,
        description="Conteúdo da mensagem (pode ser None para mensagens de sistema)"
    )
    messageType: str | None = Field(
        None,
        description="Tipo da mensagem: conversation, audioMessage, imageMessage, etc."
    )
    messageTimestamp: int | None = Field(
        None,
        description="Timestamp Unix da mensagem"
    )
    pushName: str | None = Field(
        None,
        description="Nome do remetente salvo nos contatos"
    )
    # Campos adicionais opcionais (variam por versão da Evolution API)
    status: str | None = None
    instanceId: str | None = None

    @property
    def remote_jid(self) -> str:
        """Atalho para o JID do remetente."""
        return self.key.remoteJid

    @property
    def is_from_me(self) -> bool:
        """True se a mensagem foi enviada pelo próprio bot."""
        return self.key.fromMe

    @property
    def is_group_message(self) -> bool:
        """True se a mensagem veio de um grupo (@g.us)."""
        return "@g.us" in self.key.remoteJid

    @property
    def text(self) -> str | None:
        """Atalho para extrair o texto da mensagem."""
        if self.message is None:
            return None
        return self.message.extract_text()

    @property
    def message_type(self) -> str:
        """Atalho para o tipo da mensagem."""
        if self.message is None:
            return "other"
        return self.message.get_message_type()


class EvolutionWebhookPayload(BaseModel):
    """
    Payload completo recebido no webhook da Evolution API.

    Estrutura do evento messages.upsert:
    {
        "event": "messages.upsert",
        "instance": "WHATSAPP-BAILEYS",
        "data": { ... MessageData ... },
        "destination": "5511999999999",
        "date_time": "2025-07-21T18:00:00.000Z",
        "sender": "5511999999999@s.whatsapp.net",
        "server_url": "http://evolution_api:8080",
        "apikey": "..."
    }
    """
    event: str = Field(..., description="Tipo do evento: messages.upsert, connection.update, etc.")
    instance: str = Field(..., description="Nome da instância Baileys")
    data: MessageData | dict[str, Any] = Field(
        ...,
        description="Dados do evento (estrutura varia por tipo de evento)"
    )
    destination: str | None = None
    date_time: str | None = None
    sender: str | None = None
    server_url: str | None = None
    apikey: str | None = None

    def get_message_data(self) -> MessageData | None:
        """
        Extrai o MessageData se o evento for messages.upsert.

        Returns:
            MessageData parseado ou None se não for evento de mensagem.
        """
        if self.event != "messages.upsert":
            return None

        if isinstance(self.data, MessageData):
            return self.data

        # Tenta parsear o dict como MessageData
        try:
            return MessageData.model_validate(self.data)
        except Exception:
            return None


# =============================================================================
# MODELO DE RESPOSTA DA API
# =============================================================================

class WebhookResponse(BaseModel):
    """Resposta padrão do endpoint de webhook."""
    status: Literal["ok", "ignored", "error"]
    message: str | None = None
    event: str | None = None
