# =============================================================================
# CEO Assistant — src/utils/db.py
# Repositório SQLite para Logs de Interação
#
# Por que SQLite aqui e não PostgreSQL?
# - SQLite é zero-config: sem servidor, sem credenciais, sem schema migration
# - Perfeito para logs de auditoria simples (append-only)
# - Um arquivo .db é facilmente copiado, inspecionado e backup-eado
# - Em volume Docker, persiste entre reinícios
# - Se crescer muito, migrar para PostgreSQL é simples (mesmas queries)
# =============================================================================

import aiosqlite
import structlog
from pathlib import Path

from src.config import get_settings

logger = structlog.get_logger(__name__)

# Schema da tabela de interações
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    remote_jid  TEXT    NOT NULL,
    query       TEXT    NOT NULL,
    response    TEXT    NOT NULL,
    skill       TEXT    NOT NULL DEFAULT 'desconhecido',
    data_source TEXT    NOT NULL DEFAULT 'sem dados',
    elapsed_ms  INTEGER NOT NULL DEFAULT 0,
    status      TEXT    NOT NULL DEFAULT 'success'
);

CREATE INDEX IF NOT EXISTS idx_interactions_created_at
    ON interactions (created_at);

CREATE INDEX IF NOT EXISTS idx_interactions_skill
    ON interactions (skill);
"""


async def initialize_db() -> None:
    """
    Cria o banco SQLite e as tabelas se ainda não existirem.
    Deve ser chamado no startup do FastAPI (lifespan).
    """
    settings = get_settings()
    db_path = settings.sqlite_db_path

    # Garante que o diretório existe
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_CREATE_TABLE_SQL)
        await db.commit()

    logger.info("Banco SQLite inicializado", path=db_path)


async def log_interaction(
    remote_jid: str,
    query: str,
    response: str,
    skill: str,
    data_source: str,
    elapsed_seconds: float,
    status: str = "success",
) -> int:
    """
    Registra uma interação completa no banco SQLite.

    Args:
        remote_jid: JID do usuário (mascarado para privacidade no log do Python,
                    mas armazenado completo para auditoria no banco).
        query: Pergunta original do CEO.
        response: Resposta enviada pelo assistente.
        skill: Nome da skill que processou a consulta.
        data_source: Fonte dos dados usada.
        elapsed_seconds: Tempo total de processamento em segundos.
        status: "success" ou "error".

    Returns:
        ID da linha inserida no banco.
    """
    settings = get_settings()
    elapsed_ms = round(elapsed_seconds * 1000)

    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO interactions
                (remote_jid, query, response, skill, data_source, elapsed_ms, status)
            VALUES
                (?, ?, ?, ?, ?, ?, ?)
            """,
            (remote_jid, query, response, skill, data_source, elapsed_ms, status),
        )
        await db.commit()
        row_id = cursor.lastrowid

    logger.debug(
        "Interação registrada no SQLite",
        row_id=row_id,
        skill=skill,
        elapsed_ms=elapsed_ms,
        status=status,
    )
    return row_id


async def get_recent_interactions(limit: int = 20) -> list[dict]:
    """
    Retorna as interações mais recentes para monitoramento.

    Args:
        limit: Número máximo de interações a retornar.

    Returns:
        Lista de dicts com os campos da tabela interactions.
    """
    settings = get_settings()

    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, created_at, remote_jid, query, skill, elapsed_ms, status
            FROM interactions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()

    return [dict(row) for row in rows]


async def get_stats() -> dict:
    """
    Retorna estatísticas de uso do assistente.

    Returns:
        Dict com: total_interactions, avg_elapsed_ms, skills_breakdown
    """
    settings = get_settings()

    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        # Total e média
        async with db.execute(
            "SELECT COUNT(*) as total, AVG(elapsed_ms) as avg_ms FROM interactions"
        ) as cursor:
            row = await cursor.fetchone()
            total = row[0] if row else 0
            avg_ms = round(row[1] or 0)

        # Breakdown por skill
        async with db.execute(
            """
            SELECT skill, COUNT(*) as count
            FROM interactions
            GROUP BY skill
            ORDER BY count DESC
            """
        ) as cursor:
            skills = {row[0]: row[1] for row in await cursor.fetchall()}

    return {
        "total_interactions": total,
        "avg_elapsed_ms": avg_ms,
        "skills_breakdown": skills,
    }
