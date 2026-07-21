# 🤖 CEO Assistant — Assistente Executivo via WhatsApp

> Consulte KPIs e dashboards da sua empresa em tempo real usando linguagem natural no WhatsApp.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docs.docker.com/compose/)
[![Evolution API](https://img.shields.io/badge/Evolution_API-WhatsApp-25D366?logo=whatsapp)](https://doc.evolution-api.com)

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura](#-arquitetura)
- [Estrutura de Pastas](#-estrutura-de-pastas)
- [Stack Tecnológico](#-stack-tecnológico)
- [Pré-requisitos](#-pré-requisitos)
- [Setup](#-setup)
- [Conectando o WhatsApp (QR Code)](#-conectando-o-whatsapp-qr-code)
- [Configurando o Webhook](#-configurando-o-webhook)
- [Testando](#-testando)
- [Expansão Futura](#-expansão-futura)
- [Contribuição](#-contribuição)

---

## 🎯 Visão Geral

O **CEO Assistant** é um agente de IA agentic que permite ao CEO de uma empresa consultar dados de negócio em tempo real diretamente pelo WhatsApp, usando linguagem natural:

```
CEO: "Qual foi o faturamento de julho comparado com junho?"

Bot: "📊 *Faturamento — Análise Comparativa*

Julho/2025: R$ 2.847.320,00 ↑
Junho/2025: R$ 2.610.150,00

Variação: +9,1% 🟢
Fonte: Dashboard Financeiro (atualizado em 21/07/2025 às 14h)
```

### Como funciona

1. O CEO envia uma mensagem no WhatsApp
2. A **Evolution API** (gateway Baileys) recebe e dispara um webhook para nossa aplicação
3. O **middleware de segurança** valida se o remetente é o número autorizado do CEO
4. O **agente LangGraph** interpreta a intenção e roteia para a skill correta
5. A **skill** busca os dados (via RAG + ChromaDB ou consulta direta)
6. O **LLM** formata a resposta citando fonte e período
7. A resposta é enviada de volta pelo WhatsApp
8. A interação é registrada no SQLite para auditoria

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                        DOCKER NETWORK (ceo_net)                 │
│                                                                  │
│  WhatsApp ──► Evolution API (8080) ──► FastAPI App (8000)       │
│                    │                        │                    │
│                    ▼                        ▼                    │
│               PostgreSQL              LangGraph Agent           │
│               Redis                        │                    │
│                                    ┌───────┴────────┐           │
│                                    ▼                ▼           │
│                              Skill Router      Security MW      │
│                                    │                            │
│                    ┌───────────────┼───────────────┐            │
│                    ▼               ▼               ▼            │
│             skill_financeiro  skill_vendas   skill_churn        │
│                    │               │               │            │
│                    ▼               ▼               ▼            │
│              ChromaDB (RAG)   PostgreSQL    Google Sheets       │
│                                    │                            │
│                              SQLite (logs)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Fluxo de dados detalhado

```
messages.upsert webhook
        │
        ▼
[FastAPI /webhook/evolution]
        │
        ├─ Parse payload (Pydantic)
        ├─ Validar tipo (texto/áudio/imagem)
        ├─ Extrair remoteJid + texto
        │
        ▼
[Middleware de Segurança]
        │
        ├─ Whitelist check (CEO_WHATSAPP_NUMBER)
        ├─ Rate limiting (Redis)
        │
        ▼
[LangGraph Agent — Grafo de Estados]
        │
        ├─ Nó: interpret_intent (LLM classifica a intenção)
        ├─ Nó: route_skill (decide qual skill chamar)
        ├─ Nó: execute_skill (busca dados + RAG)
        ├─ Nó: format_response (LLM formata resposta final)
        ├─ Nó: send_message (Evolution API /message/sendText)
        └─ Nó: log_interaction (SQLite)
```

---

## 📁 Estrutura de Pastas

```
ceo-assistant/
│
├── docker-compose.yml          # Orquestra Evolution API + Postgres + Redis + App
├── Dockerfile                  # Imagem da aplicação FastAPI
├── requirements.txt            # Dependências Python fixadas
├── pyproject.toml              # Configuração pytest + ferramentas
├── .env.example                # Template de variáveis de ambiente
├── .gitignore
├── README.md
│
├── secrets/                    # ⚠️ NÃO commitado — credenciais sensíveis
│   └── google_credentials.json # Service Account do Google Sheets
│
└── src/                        # Código-fonte da aplicação
    ├── __init__.py
    ├── main.py                 # Entrypoint FastAPI
    ├── config.py               # Pydantic Settings (carrega .env)
    │
    ├── api/                    # Camada HTTP
    │   ├── __init__.py
    │   ├── webhook.py          # Router: recebe eventos da Evolution API
    │   ├── evolution.py        # Cliente HTTP para a Evolution API
    │   ├── security.py         # Whitelist + rate limiting
    │   └── models.py           # Pydantic models do payload Evolution API
    │
    ├── core/                   # Núcleo do agente
    │   ├── __init__.py
    │   ├── agent.py            # Grafo LangGraph (orquestrador principal)
    │   ├── router.py           # Roteador de skills por intenção
    │   ├── rag.py              # Camada RAG: indexação + busca ChromaDB
    │   └── llm.py              # Factory: retorna LLM correto (Ollama/OpenAI/Claude)
    │
    ├── skills/                 # Módulos de domínio (independentes)
    │   ├── __init__.py
    │   ├── skill_financeiro.py # Faturamento, receita, custos, margem
    │   ├── skill_vendas.py     # Volume, ticket médio, conversão
    │   └── skill_churn.py      # Taxa de cancelamento, LTV, clientes em risco
    │
    ├── connectors/             # Adaptadores para fontes de dados
    │   ├── __init__.py
    │   ├── sheets.py           # Google Sheets via Service Account
    │   ├── postgres.py         # PostgreSQL direto
    │   └── rest_api.py         # APIs REST genéricas (BI externo)
    │
    └── utils/                  # Utilitários transversais
        ├── __init__.py
        ├── logger.py           # Logging estruturado (structlog → JSON)
        └── db.py               # Repositório SQLite para logs de interação

tests/                          # Suite pytest
    ├── __init__.py
    ├── test_webhook.py         # Parsing de payload Evolution API
    ├── test_router.py          # Roteamento de skills por intenção
    └── test_skill_financeiro.py # Skill financeira com dados fictícios
```

---

## 🛠️ Stack Tecnológico

| Camada | Tecnologia | Justificativa |
|--------|------------|---------------|
| Canal WhatsApp | Evolution API (Baileys) | Self-hosted, sem custos de API comercial |
| Web Framework | FastAPI + Uvicorn | Async nativo, validação Pydantic, alta performance |
| Orquestração | LangGraph | Grafo de estados explícito, melhor controle de fluxo que chains simples |
| LLM | Ollama / OpenAI / Claude | Configurável; Ollama para dev local sem custo |
| RAG | LangChain + ChromaDB | Indexação semântica local, sem dependência de serviço externo |
| Embeddings | sentence-transformers | Modelos locais de alta qualidade (all-MiniLM-L6-v2) |
| Cache | Redis | Rate limiting e cache de sessão de alta performance |
| Logs | SQLite (aiosqlite) | Leve, zero-config, perfeito para auditoria local |
| Dados | Google Sheets / PostgreSQL / REST | Flexível: conecta às fontes existentes da empresa |

---

## ✅ Pré-requisitos

- **Docker** e **Docker Compose** instalados
- **Git**
- Número de WhatsApp dedicado para o bot (chip separado recomendado)
- (Opcional) **Ollama** instalado no host para LLM local

---

## 🚀 Setup

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/ceo-assistant.git
cd ceo-assistant
```

### 2. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` com seus valores reais:

```env
EVOLUTION_API_KEY=sua_chave_secreta_segura
CEO_WHATSAPP_NUMBER=5511999999999@s.whatsapp.net
LLM_PROVIDER=ollama          # ou openai, anthropic
LLM_MODEL=llama3
```

### 3. Suba os containers

```bash
docker-compose up -d
```

Aguarde todos os serviços ficarem saudáveis:

```bash
docker-compose ps
```

Saída esperada:
```
NAME              STATUS         PORTS
ceo_postgres      healthy        5432/tcp
ceo_redis         healthy        6379/tcp
ceo_evolution     running        0.0.0.0:8080->8080/tcp
ceo_assistant     healthy        0.0.0.0:8000->8000/tcp
```

### 4. Verifique os logs

```bash
# Todos os serviços
docker-compose logs -f

# Apenas a aplicação
docker-compose logs -f ceo_app
```

---

## 📱 Conectando o WhatsApp (QR Code)

### Passo 1: Crie a instância na Evolution API

```bash
curl -X POST http://localhost:8080/instance/create \
  -H "apikey: sua_chave_secreta_segura" \
  -H "Content-Type: application/json" \
  -d '{
    "instanceName": "WHATSAPP-BAILEYS",
    "qrcode": true,
    "integration": "WHATSAPP-BAILEYS"
  }'
```

### Passo 2: Obtenha o QR Code

```bash
curl http://localhost:8080/instance/connect/WHATSAPP-BAILEYS \
  -H "apikey: sua_chave_secreta_segura"
```

A resposta contém o QR Code em base64. Para visualizá-lo:

**Opção A — via Manager Web (mais fácil):**
Acesse `http://localhost:8080/manager` no navegador. Clique em sua instância e escaneie o QR Code com o WhatsApp do número do bot.

**Opção B — salvar QR Code como imagem:**
```bash
# A resposta JSON tem o campo "base64" — salve como PNG
curl http://localhost:8080/instance/connect/WHATSAPP-BAILEYS \
  -H "apikey: sua_chave_secreta_segura" \
  | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
qr = data.get('base64', '').replace('data:image/png;base64,', '')
with open('qrcode.png', 'wb') as f:
    f.write(base64.b64decode(qr))
print('QR Code salvo em qrcode.png')
"
```

### Passo 3: Escaneie com o WhatsApp

1. Abra o WhatsApp no chip dedicado ao bot
2. Vá em **Configurações → Aparelhos conectados → Conectar aparelho**
3. Escaneie o QR Code gerado
4. Aguarde a confirmação de conexão

### Verificar status da conexão

```bash
curl http://localhost:8080/instance/connectionState/WHATSAPP-BAILEYS \
  -H "apikey: sua_chave_secreta_segura"
```

Resposta esperada: `{"state": "open"}`

---

## 🔗 Configurando o Webhook

Após a instância estar conectada, configure o webhook para apontar para nossa aplicação:

```bash
curl -X POST http://localhost:8080/webhook/set/WHATSAPP-BAILEYS \
  -H "apikey: sua_chave_secreta_segura" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://ceo_assistant:8000/webhook/evolution",
    "webhook_by_events": false,
    "webhook_base64": false,
    "events": [
      "MESSAGES_UPSERT"
    ]
  }'
```

> **Nota:** Em produção, use a URL pública do seu servidor ou um túnel ngrok:
> ```bash
> ngrok http 8000
> # Use a URL https gerada no campo "url" acima
> ```

---

## 🧪 Testando

### Testes unitários

```bash
# Instalar dependências localmente (para desenvolvimento)
pip install -r requirements.txt

# Rodar todos os testes com cobertura
pytest

# Rodar um arquivo específico
pytest tests/test_webhook.py -v
```

### Teste manual do webhook (curl)

```bash
# Simula um payload messages.upsert da Evolution API
curl -X POST http://localhost:8000/webhook/evolution \
  -H "Content-Type: application/json" \
  -d '{
    "event": "messages.upsert",
    "instance": "WHATSAPP-BAILEYS",
    "data": {
      "key": {
        "remoteJid": "5511999999999@s.whatsapp.net",
        "fromMe": false,
        "id": "MSGID123"
      },
      "message": {
        "conversation": "Qual foi o faturamento de julho?"
      },
      "messageType": "conversation",
      "messageTimestamp": 1721563200
    }
  }'
```

### Healthcheck

```bash
curl http://localhost:8000/health
# {"status": "ok", "service": "CEO Assistant"}
```

---

## 🔮 Expansão Futura

| Feature | Dificuldade | Descrição |
|---------|-------------|-----------|
| 📊 Gráficos inline | Média | Gerar gráficos matplotlib e enviar como imagem via WhatsApp |
| 🔔 Alertas proativos | Média | Monitorar KPIs e notificar o CEO quando metas forem atingidas/perdidas |
| 🎤 Mensagem de voz | Alta | Transcrever áudio com Whisper e processar a pergunta |
| 📅 Agendamento | Baixa | Enviar relatório diário automático todo dia útil às 8h |
| 👥 Multi-usuário | Média | Expandir whitelist para equipe executiva com roles diferentes |
| 🔄 Memória de conversa | Média | Manter contexto das últimas N mensagens por usuário |
| 📈 Dashboard web | Alta | Interface web para visualizar logs de interações e KPIs |
| 🌐 Multi-instância | Alta | Suporte a múltiplas empresas na mesma infraestrutura |

---

## 🤝 Contribuição

Projeto de portfólio desenvolvido por **Erick Vinicius** — estudante de IA/ML.

- LinkedIn: [linkedin.com/in/seu-perfil](https://linkedin.com/in/)
- GitHub: [github.com/seu-usuario](https://github.com/)

---

## 📄 Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.
