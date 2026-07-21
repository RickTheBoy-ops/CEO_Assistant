# =============================================================================
# CEO Assistant — src/api/security.py
# Middleware de Segurança: Whitelist + Rate Limiting
#
# Por que duas camadas de segurança?
# - Whitelist: garante que APENAS o CEO pode usar o bot — qualquer outro
#   número que acesse a instância é silenciosamente ignorado.
# - Rate Limiting: protege contra loop de mensagens (o próprio bot respondendo
#   a si mesmo), scripts maliciosos, ou o CEO mandando muitas msgs seguidas.
#
# Por que Redis para rate limiting (e não in-memory)?
# - In-memory seria resetado a cada restart do container
# - Redis persiste os contadores entre restarts
# - Redis suporta expiração automática de chaves (EXPIRE) — ideal para janelas
# - Em cenários multi-worker, o Redis é compartilhado entre todos os workers
# =============================================================================

import structlog
from redis import Redis
from redis.exceptions import RedisError

from src.config import get_settings

logger = structlog.get_logger(__name__)

# Prefixo das chaves de rate limiting no Redis
_RATE_LIMIT_PREFIX = "ceo_assistant:rate_limit"


def get_redis_client() -> Redis:
    """
    Cria e retorna cliente Redis.
    Em caso de falha de conexão, loga o erro mas não bloqueia a aplicação
    (fail-open: prefere processar a mensagem a bloquear o CEO).
    """
    settings = get_settings()
    try:
        client = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()  # valida a conexão
        return client
    except RedisError as exc:
        logger.error("Falha ao conectar ao Redis", error=str(exc))
        raise


class SecurityMiddleware:
    """
    Validações de segurança para mensagens recebidas via webhook.

    Responsabilidades:
    1. Verificar se o remetente está na whitelist (número do CEO)
    2. Verificar se a mensagem não é do próprio bot (evita loop)
    3. Aplicar rate limiting por número de telefone
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis: Redis | None = None

    def _get_redis(self) -> Redis | None:
        """Retorna cliente Redis com lazy initialization."""
        if self._redis is None:
            try:
                self._redis = get_redis_client()
            except Exception:
                logger.warning(
                    "Redis indisponível — rate limiting desativado para esta requisição"
                )
                return None
        return self._redis

    def is_authorized(self, remote_jid: str) -> bool:
        """
        Verifica se o remetente está autorizado a usar o assistente.

        A comparação é normalizada para ser case-insensitive e
        tolerante a diferenças de formatação.

        Args:
            remote_jid: JID do remetente (ex: 5511999999999@s.whatsapp.net)

        Returns:
            True se o remetente for o CEO autorizado, False caso contrário.
        """
        authorized = self._settings.ceo_whatsapp_number.strip().lower()
        incoming = remote_jid.strip().lower()

        is_ok = incoming == authorized

        if not is_ok:
            logger.warning(
                "Acesso negado — número não autorizado",
                remote_jid=self._mask_jid(remote_jid),
                authorized_mask=self._mask_jid(authorized),
            )

        return is_ok

    def is_rate_limited(self, remote_jid: str) -> bool:
        """
        Verifica se o número excedeu o limite de requisições.

        Implementa o algoritmo "Fixed Window Counter":
        - Cada número tem um contador no Redis com TTL = RATE_LIMIT_WINDOW_SECONDS
        - Se o contador ultrapassar RATE_LIMIT_REQUESTS, rejeita até o TTL expirar
        - O contador é criado automaticamente na primeira mensagem da janela

        Args:
            remote_jid: JID do remetente.

        Returns:
            True se o número estiver acima do limite (deve ser bloqueado).
        """
        redis = self._get_redis()
        if redis is None:
            # Se Redis está fora, não aplica rate limiting (fail-open)
            return False

        key = f"{_RATE_LIMIT_PREFIX}:{remote_jid}"
        limit = self._settings.rate_limit_requests
        window = self._settings.rate_limit_window_seconds

        try:
            # Pipeline para garantir atomicidade das operações INCR + EXPIRE
            with redis.pipeline() as pipe:
                pipe.incr(key)
                pipe.expire(key, window)
                results = pipe.execute()

            current_count = results[0]

            if current_count > limit:
                logger.warning(
                    "Rate limit atingido",
                    remote_jid=self._mask_jid(remote_jid),
                    count=current_count,
                    limit=limit,
                    window_seconds=window,
                )
                return True

            logger.debug(
                "Rate limit OK",
                remote_jid=self._mask_jid(remote_jid),
                count=current_count,
                limit=limit,
            )
            return False

        except RedisError as exc:
            logger.error(
                "Erro no Redis durante rate limiting, permitindo requisição",
                error=str(exc),
            )
            return False  # fail-open

    def validate_message(
        self,
        remote_jid: str,
        is_from_me: bool,
        is_group: bool,
        message_type: str,
    ) -> tuple[bool, str]:
        """
        Validação completa de uma mensagem recebida.

        Args:
            remote_jid: JID do remetente.
            is_from_me: True se a mensagem foi enviada pelo bot.
            is_group: True se é mensagem de grupo.
            message_type: Tipo da mensagem ("text", "audio", "image", etc.).

        Returns:
            Tupla (deve_processar: bool, motivo_rejeicao: str)
        """
        # 1. Ignora mensagens enviadas pelo próprio bot (evita loop infinito)
        if is_from_me:
            return False, "Mensagem própria ignorada (fromMe=True)"

        # 2. Ignora mensagens de grupos (bot é pessoal do CEO)
        if is_group:
            return False, "Mensagens de grupos não são suportadas"

        # 3. Ignora tipos de mensagem não suportados
        supported_types = {"text"}
        if message_type not in supported_types:
            logger.info(
                "Tipo de mensagem não suportado ignorado",
                message_type=message_type,
                remote_jid=self._mask_jid(remote_jid),
            )
            return False, f"Tipo '{message_type}' não suportado (apenas texto)"

        # 4. Verifica whitelist
        if not self.is_authorized(remote_jid):
            return False, "Número não autorizado"

        # 5. Verifica rate limiting
        if self.is_rate_limited(remote_jid):
            return False, "Rate limit excedido"

        return True, "OK"

    @staticmethod
    def _mask_jid(jid: str) -> str:
        """
        Mascara o JID para logging seguro.
        Ex: 5511999999999@s.whatsapp.net → 551199****999@s.whatsapp.net
        """
        if "@" not in jid:
            return "***"
        number, domain = jid.split("@", 1)
        if len(number) > 6:
            masked = number[:6] + "****" + number[-3:]
        else:
            masked = "***"
        return f"{masked}@{domain}"


# Instância singleton
_security = SecurityMiddleware()


def get_security() -> SecurityMiddleware:
    """Retorna instância singleton do SecurityMiddleware."""
    return _security
