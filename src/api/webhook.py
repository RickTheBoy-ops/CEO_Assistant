# =============================================================================
# CEO Assistant — src/api/webhook.py
# Router FastAPI para Recebimento de Eventos da Evolution API
#
# Este módulo implementa o endpoint que recebe todos os eventos da Evolution API
# via webhook (mensagens, atualizações de conexão, etc.) e despacha para o agente.
#
# Por que processar o agente em background (asyncio.create_task)?
# A Evolution API aguarda a resposta HTTP do webhook. Se demorarmos a responder
# (o LLM pode levar 5-30 segundos), o webhook pode timeout e reenviar a mensagem.
# Respondemos 200 OK imediatamente e processamos o agente em background.
# =============================================================================

import asyncio
import structlog
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.models import EvolutionWebhookPayload, WebhookResponse
from src.api.security import get_security
from src.api.evolution import send_text_message

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/webhook", tags=["Webhook"])


@router.post(
    "/evolution",
    response_model=WebhookResponse,
    summary="Endpoint de webhook da Evolution API",
    description="Recebe eventos messages.upsert e processa mensagens do CEO.",
)
async def receive_evolution_webhook(
    request: Request,
    payload: EvolutionWebhookPayload,
) -> JSONResponse:
    """
    Endpoint principal de webhook.

    Fluxo:
    1. Recebe e valida o payload da Evolution API (Pydantic)
    2. Filtra eventos que não são messages.upsert
    3. Extrai dados da mensagem (remoteJid, texto, tipo)
    4. Valida segurança (whitelist + rate limiting)
    5. Responde 200 OK imediatamente (evita timeout do webhook)
    6. Processa o agente em background (asyncio.create_task)
    """
    log = logger.bind(event=payload.event, instance=payload.instance)

    # --- Filtra eventos irrelevantes ---
    if payload.event != "messages.upsert":
        log.debug("Evento ignorado (não é messages.upsert)")
        return JSONResponse(
            content=WebhookResponse(
                status="ignored",
                event=payload.event,
                message="Evento não processado",
            ).model_dump()
        )

    # --- Extrai dados da mensagem ---
    message_data = payload.get_message_data()
    if message_data is None:
        log.warning("Falha ao parsear MessageData do payload")
        return JSONResponse(
            content=WebhookResponse(
                status="ignored",
                message="Payload de mensagem inválido",
            ).model_dump()
        )

    remote_jid = message_data.remote_jid
    message_type = message_data.message_type
    text = message_data.text
    is_from_me = message_data.is_from_me
    is_group = message_data.is_group_message

    log = log.bind(
        remote_jid_masked=_mask_jid(remote_jid),
        message_type=message_type,
        is_from_me=is_from_me,
        is_group=is_group,
    )

    # --- Validação de segurança ---
    security = get_security()
    should_process, reason = security.validate_message(
        remote_jid=remote_jid,
        is_from_me=is_from_me,
        is_group=is_group,
        message_type=message_type,
    )

    if not should_process:
        log.info("Mensagem descartada", reason=reason)
        return JSONResponse(
            content=WebhookResponse(
                status="ignored",
                message=reason,
            ).model_dump()
        )

    # --- Valida texto ---
    if not text:
        log.info("Mensagem sem texto válido ignorada")
        return JSONResponse(
            content=WebhookResponse(
                status="ignored",
                message="Mensagem sem conteúdo de texto",
            ).model_dump()
        )

    log.info("Mensagem aceita para processamento", text_preview=text[:60])

    # --- Responde 200 OK imediatamente e processa em background ---
    asyncio.create_task(
        _process_message_background(
            remote_jid=remote_jid,
            text=text,
            log=log,
        )
    )

    return JSONResponse(
        content=WebhookResponse(
            status="ok",
            message="Mensagem recebida, processando...",
        ).model_dump()
    )


async def _process_message_background(
    remote_jid: str,
    text: str,
    log: structlog.BoundLogger,
) -> None:
    """
    Processa a mensagem do CEO em background.
    Executa o agente LangGraph e trata erros de forma segura.

    Esta função nunca deve lançar exceções não tratadas (asyncio.Task
    ignora exceções silenciosamente se não houver handler).
    """
    # Import lazy para evitar circular import e permitir startup rápido
    from src.core.agent import run_agent

    try:
        log.info("Iniciando processamento em background")

        # Envia indicador de "digitando..." para melhor UX
        try:
            from src.api.evolution import get_evolution_client
            client = get_evolution_client()
            await client.send_typing_indicator(to=remote_jid)
        except Exception:
            pass  # não crítico

        # Executa o agente completo
        response = await run_agent(query=text, remote_jid=remote_jid)

        log.info(
            "Processamento concluído",
            response_preview=response[:60] if response else "vazio",
        )

    except Exception as exc:
        log.error(
            "Erro crítico no processamento em background",
            error=str(exc),
            exc_info=True,
        )
        # Tenta enviar mensagem de erro para o CEO
        try:
            await send_text_message(
                to=remote_jid,
                text="❌ Ocorreu um erro ao processar sua consulta. Por favor, tente novamente.",
            )
        except Exception:
            log.error("Falha ao enviar mensagem de erro")


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
