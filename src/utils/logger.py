# =============================================================================
# CEO Assistant — src/utils/logger.py
# Logging Estruturado com structlog
#
# Por que structlog em vez do logging padrão?
# - Output em JSON nativo: fácil ingestão por ELK, Loki, CloudWatch, etc.
# - Contexto estruturado: campos chave=valor em vez de strings concatenadas
# - Performance: lazy evaluation dos campos (não processa se log descartado)
# - Thread-safe e async-friendly
# =============================================================================

import logging
import structlog
from src.config import get_settings


def setup_logging() -> None:
    """
    Configura o logging estruturado para toda a aplicação.
    Deve ser chamado UMA vez na inicialização do FastAPI (lifespan).
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configura o logging padrão do Python para capturar libs que não usam structlog
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )

    # Silencia logs verbosos de libs externas
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Processadores do structlog (pipeline de transformação de cada log)
    processors = [
        structlog.contextvars.merge_contextvars,         # contexto de request
        structlog.stdlib.add_log_level,                  # nível do log
        structlog.stdlib.add_logger_name,                # nome do logger
        structlog.processors.TimeStamper(fmt="iso"),     # timestamp ISO 8601
        structlog.processors.StackInfoRenderer(),        # stack info
        structlog.processors.format_exc_info,            # formatação de exceções
    ]

    # Em produção, usa JSON puro; em desenvolvimento, usa formato colorido
    if settings.log_level == "DEBUG":
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
