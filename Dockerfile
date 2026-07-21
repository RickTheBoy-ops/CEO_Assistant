# =============================================================================
# CEO Assistant — Dockerfile
# Imagem da aplicação FastAPI (webhook receiver + agente LangGraph)
# =============================================================================

# Usa Python 3.11 slim para imagem leve e compatível com todas as dependências
FROM python:3.11-slim

# Metadados da imagem
LABEL maintainer="Erick Vinicius"
LABEL description="CEO Assistant — Agente executivo via WhatsApp com LangGraph + RAG"
LABEL version="1.0.0"

# Variáveis de ambiente para comportamento correto do Python em containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências de sistema necessárias para algumas libs Python
# libpq-dev: para psycopg2 (PostgreSQL)
# curl: para healthcheck e debug
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python primeiro (melhor cache de layers Docker)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o código-fonte da aplicação
COPY src/ ./src/

# Cria os diretórios de dados que serão sobrescritos pelos volumes
RUN mkdir -p /data/chroma /data/sqlite /app/secrets

# Usuário não-root por segurança
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app /data
USER appuser

# Porta que o Uvicorn vai escutar
EXPOSE 8000

# Healthcheck: verifica se a API está respondendo
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicialização
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
