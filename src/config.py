# =============================================================================
# CEO Assistant — Configurações centralizadas com Pydantic Settings
#
# Por que Pydantic Settings?
# - Carrega variáveis do .env automaticamente com type hints e validação
# - Evita os.getenv() espalhados pelo código
# - Facilita testes (injeção de configuração via parâmetros)
# =============================================================================

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações globais da aplicação carregadas do .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignora variáveis no .env não declaradas aqui
    )

    # --- Evolution API ---
    evolution_api_key: str = Field(..., description="Chave de autenticação global da Evolution API")
    evolution_base_url: str = Field("http://evolution_api:8080", description="URL base da Evolution API")
    evolution_instance_name: str = Field("WHATSAPP-BAILEYS", description="Nome da instância Baileys")

    # --- Segurança ---
    ceo_whatsapp_number: str = Field(..., description="remoteJid autorizado (ex: 5511999999999@s.whatsapp.net)")

    # --- LLM ---
    llm_provider: Literal["ollama", "openai", "anthropic"] = Field("ollama", description="Provedor do LLM")
    llm_model: str = Field("llama3", description="Nome do modelo")
    ollama_base_url: str = Field("http://host.docker.internal:11434")
    openai_api_key: str = Field("", description="Chave OpenAI (opcional)")
    anthropic_api_key: str = Field("", description="Chave Anthropic (opcional)")

    # --- ChromaDB ---
    chroma_persist_dir: str = Field("/data/chroma", description="Diretório de persistência do ChromaDB")

    # --- SQLite (logs) ---
    sqlite_db_path: str = Field("/data/sqlite/interactions.db")

    # --- Redis ---
    redis_url: str = Field("redis://:redis_pass@redis:6379/0")

    # --- Rate Limiting ---
    rate_limit_requests: int = Field(10, description="Máx. requisições por janela")
    rate_limit_window_seconds: int = Field(60, description="Janela de tempo em segundos")

    # --- Google Sheets ---
    google_sheets_credentials_path: str = Field("/app/secrets/google_credentials.json")
    google_sheets_spreadsheet_id: str = Field("", description="ID da planilha de dados")

    # --- App ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retorna instância singleton das configurações.
    O @lru_cache garante que o .env seja lido apenas uma vez.
    """
    return Settings()
