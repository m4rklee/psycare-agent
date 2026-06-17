# PsyCare Agent

> A multimodal mental health support agent system for campus psychological care.

PsyCare Agent 是一个面向校园心理健康场景的多模态 Agent 系统。项目基于 Python/FastAPI 构建，结合多 Agent 编排、ReAct 工具调用、校园心理知识库检索、短期/长期记忆、SSE 流式聊天、风险报告与预警流程，为学生端提供心理支持型 AI 对话体验。

本项目定位为心理健康支持与工程原型，不替代专业心理咨询、医学诊断或紧急救援服务。

## Highlights

- **Multi-Agent Swarm**: LeadAgent 负责任务分解、Worker Agent 分派和结果汇总。
- **ReAct Worker Loop**: Consultation、Diagnostic、Research Agent 可按需调用 skills 并观察结果。
- **Campus Mental Health Skills**: 内置知识检索、风险评估、症状模式分析、生活支持建议、历史检索等 9 个 skills。
- **Streaming Chat UI**: 后端通过 SSE 输出 token，前端实时渲染主回复和旁路旧链路观察。
- **Memory & Session Recovery**: 支持会话历史、长期摘要、任务状态记录和会话恢复。
- **Risk Report Workflow**: 高风险场景可生成心理报告，并通过 Excel、邮件或 HTTP/MCP 链路触发预警。
- **Multimodal Pipeline**: 支持语音、图像/视频情绪线索和文本输入的融合分析。

## Demo

项目提供静态 Web 聊天页面，默认地址：

```text
http://localhost:8080
```

页面支持学生聊天、管理员报告查看、旁路旧链路回复展示、麦克风录音和多模态输入。模型权重较大，未随仓库提交；模型导入方式见 [LoRA fine-tuning guide](docs/qwen25-7b-lora-finetune-guide.md)。

## Architecture

```text
Frontend / SSE Chat UI
        |
        v
FastAPI ChatService
        |
        +--> Legacy sidecar response
        |
        v
LeadAgent
        |
        +--> ConsultationAgent
        +--> DiagnosticAgent
        +--> ResearchAgent
        |
        v
Skills / Tools / RAG / Memory / MCP
        |
        v
Final response / report / alert
```

Core flow:

1. 用户在学生端发送文本、语音或多模态消息。
2. 后端恢复会话上下文，写入用户消息和多模态 system memory。
3. LeadAgent 用 LLM 分解任务，并分派给一个或多个 Worker Agent。
4. Worker Agent 通过 ReAct 循环选择 skills/tools，执行观察后生成结构化回复。
5. LeadAgent 对多 Agent 结果进行汇总；单 Agent 成功时可直接返回 Worker 回复。
6. ChatService 将主回复通过 SSE 真实流式发送到前端，并保存为会话 assistant 消息。

## Features

| Module | Description |
|---|---|
| **Agent Orchestration** | LeadAgent 按复杂度分派 Consultation、Diagnostic、Research Agent。 |
| **ReAct Agent Loop** | Worker Agent 使用 JSON ReAct 协议执行 action/observation/final。 |
| **Capability Layer** | Skills 可自动调用；MCP/tools 进入统一注册结构，默认阻断高副作用自动执行。 |
| **Knowledge Retrieval** | 内置校园心理健康知识库，可选 Chroma 检索。 |
| **Memory System** | 短期记忆适配会话历史，长期记忆保存摘要并使用 hash 去重。 |
| **Risk Workflow** | 高风险输入可触发报告、Excel 写入和邮件/HTTP/MCP 预警。 |
| **Compatibility Layer** | 旧聊天链路作为旁路观察展示，便于对比迁移效果。 |

## Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy, Alembic, Pydantic
- **Agent Runtime**: Custom Lead/Worker Agent framework, ReAct loop, skills registry
- **LLM Providers**: Ollama, OpenAI-compatible API, mock client for offline tests
- **Storage**: MySQL, Redis, SQLite for local smoke tests
- **Knowledge/RAG**: Local retrieval, optional Chroma
- **Frontend**: Static HTML/CSS/JavaScript, SSE streaming
- **Multimodal**: Whisper integration, POSTER/UniFER/Fed-PSYAU experiment labs
- **Deployment**: Docker Compose, uv, local shell scripts

## Quick Start

Python 版使用 `uv` 管理依赖。默认连接 MySQL、Redis，并通过 Ollama 调用本地模型。

```bash
git clone https://github.com/m4rklee/psycare-agent.git
cd psycare-agent
docker compose up -d mysql redis chroma mailpit
./scripts/run-python-dev.sh
```

打开：

```text
http://localhost:8080
```

首次启动会创建两个默认账号：

```text
admin / admin123
student / student123
```

如果没有 MySQL 或本地模型，只想离线烟测完整网页和接口：

```bash
DB_URL="sqlite+aiosqlite:///./data/python-dev.db" \
AI_PROVIDER=mock \
USE_CHROMA=false \
MCP_EXCEL_MODE=local \
MCP_EMAIL_MODE=log \
uv run uvicorn app.main:app --host 127.0.0.1 --port 8080
```

## Usage

学生端流式聊天：

```bash
curl -N -u student:student123 \
  -H 'Content-Type: application/json' \
  -d '{"message":"我最近很焦虑，晚上总是睡不着"}' \
  http://localhost:8080/api/chat/stream
```

高风险样例会触发报告与预警流程：

```bash
curl -N -u student:student123 \
  -H 'Content-Type: application/json' \
  -d '{"message":"我不想活了，感觉撑不下去了"}' \
  http://localhost:8080/api/chat/stream
```

管理员查看后台报告：

```bash
curl -u admin:admin123 http://localhost:8080/api/admin/reports
```

查看当前模型与服务状态：

```bash
curl -u student:student123 http://localhost:8080/api/agent/status
```

## Project Structure

```text
app/
  agent/                 # Multi-agent framework, ReAct runtime, memory, skills loader
  api/                   # Chat / admin / MCP / profile / status APIs
  core/                  # Settings, database, auth, migrations
  models/                # SQLAlchemy entities and enums
  schemas/               # Pydantic request/response schemas
  services/              # AI clients, chat flow, knowledge, multimodal, reports, tools
  static/                # Static web chat UI
  knowledge/             # Campus mental health knowledge base
skills/                  # External skill definitions and async handlers
alembic/                 # Database migrations
tests/                   # Python regression and contract tests
experiments/             # Multimodal model labs and demo servers
models/                  # Model metadata and Modelfiles; weights are not committed
```

## Model Setup

默认 Ollama 模型名：

```text
multimodalAgent-qwen2.5-7b-ft:latest
```

模型权重不会提交到 GitHub。首次运行或重新导入本地 GGUF 模型时执行：

```bash
./scripts/create-finetuned-model.sh
```

也可以切换 OpenAI-compatible provider：

```bash
AI_PROVIDER=openai \
OPENAI_API_KEY=your_api_key \
OPENAI_MODEL=gpt-4o-mini \
uv run uvicorn app.main:app --host 127.0.0.1 --port 8080
```

## Tests

```bash
uv run pytest -q
```

当前测试覆盖 ChatService、Agent framework、MCP 工具、知识库、规则评估、多模态实验服务和应用契约。

## Safety & Limitations

- 本系统只用于校园心理健康支持、科普和工程研究，不替代专业心理咨询、医学诊断或紧急救助。
- 高风险场景应联系学校心理中心、辅导员、可信任的人或当地紧急救援服务。
- MCP/tools 默认不全面放开自动执行，避免 Agent 自主触发高副作用操作。
- 本仓库不包含本地大模型权重、运行数据库、Excel 报告和 `.env`。

## Roadmap

- Improve long-term memory retrieval and cross-session case search.
- Add richer observability for ReAct traces and skill results.
- Expand campus mental health knowledge evaluation.
- Harden MCP tool permission policies.
- Add production deployment and monitoring guide.
