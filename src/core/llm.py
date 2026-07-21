# =============================================================================
# CEO Assistant — src/core/llm.py
# Factory de LLM configurável por provedor
#
# Por que Factory Pattern aqui?
# O agente LangGraph precisa de uma instância BaseChatModel independente do
# provedor concreto (Ollama, OpenAI, Claude). A factory centraliza essa
# decisão em um único lugar, permitindo trocar o provedor apenas no .env
# sem alterar nenhum outro módulo do sistema.
# =============================================================================

import structlog
from functools import lru_cache
from langchain_core.language_models import BaseChatModel

from src.config import get_settings

logger = structlog.get_logger(__name__)


def create_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Cria e retorna a instância do LLM configurada via variável LLM_PROVIDER.

    Args:
        temperature: Temperatura do modelo (0.0 = determinístico, ideal para
                     consultas de dados onde precisão > criatividade).

    Returns:
        Instância de BaseChatModel compatível com LangChain/LangGraph.

    Raises:
        ValueError: Se o provedor configurado não for suportado.
        ImportError: Se a biblioteca do provedor não estiver instalada.
    """
    settings = get_settings()
    provider = settings.llm_provider
    model = settings.llm_model

    logger.info("Inicializando LLM", provider=provider, model=model)

    if provider == "ollama":
        return _create_ollama(model=model, temperature=temperature)
    elif provider == "openai":
        return _create_openai(model=model, temperature=temperature)
    elif provider == "anthropic":
        return _create_anthropic(model=model, temperature=temperature)
    else:
        raise ValueError(
            f"Provedor LLM não suportado: '{provider}'. "
            "Use: 'ollama', 'openai' ou 'anthropic'."
        )


def _create_ollama(model: str, temperature: float) -> BaseChatModel:
    """
    Cria cliente Ollama via LangChain.

    Por que Ollama para desenvolvimento?
    - Zero custo: roda localmente no host
    - Privacidade: dados não saem da máquina
    - Offline: funciona sem internet
    - Modelos: llama3, mistral, gemma2, phi3, qwen2.5
    """
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-community não instalado. "
            "Execute: pip install langchain-community"
        )

    settings = get_settings()
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        # Timeout maior para modelos maiores (llama3:70b pode ser lento)
        timeout=120,
        # Mantém o modelo carregado na memória por 5 minutos após uso
        keep_alive="5m",
    )


def _create_openai(model: str, temperature: float) -> BaseChatModel:
    """
    Cria cliente OpenAI via LangChain.

    Modelos recomendados:
    - gpt-4o-mini: mais rápido e barato, ótimo para consultas estruturadas
    - gpt-4o: melhor raciocínio, para análises complexas
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai não instalado. "
            "Execute: pip install langchain-openai"
        )

    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY não configurada no .env. "
            "Configure a chave ou use LLM_PROVIDER=ollama."
        )

    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=temperature,
        max_retries=3,
        request_timeout=60,
    )


def _create_anthropic(model: str, temperature: float) -> BaseChatModel:
    """
    Cria cliente Anthropic/Claude via LangChain.

    Modelos recomendados:
    - claude-3-5-haiku-20241022: mais rápido, ideal para consultas de dados
    - claude-3-5-sonnet-20241022: melhor raciocínio e formatação de texto
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic não instalado. "
            "Execute: pip install langchain-anthropic"
        )

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY não configurada no .env. "
            "Configure a chave ou use LLM_PROVIDER=ollama."
        )

    return ChatAnthropic(
        model=model,
        api_key=settings.anthropic_api_key,
        temperature=temperature,
        max_retries=3,
        timeout=60,
    )


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Retorna instância singleton do LLM.
    Usa cache para evitar reinicializar o cliente a cada requisição.
    """
    return create_llm(temperature=0.0)


@lru_cache(maxsize=1)
def get_creative_llm() -> BaseChatModel:
    """
    Retorna instância do LLM com temperatura mais alta para respostas
    mais naturais e elaboradas (usado no nó format_response do agente).
    """
    return create_llm(temperature=0.3)
