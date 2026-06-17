# PsyCare Agent

> PsyCare Agent 是一个面向校园心理健康场景的多模态 Agent 系统。

PsyCare Agent 是一个面向校园心理健康场景的多模态 Agent 系统。项目基于 Python/FastAPI 构建，结合多 Agent 编排、ReAct 工具调用、校园心理知识库检索、短期/长期记忆、SSE 流式聊天、风险报告与预警流程，为学生端提供心理支持型 AI 对话体验。

## 功能介绍

- **ReAct智能体范式**：基于ReAct智能体框架实现Think-Act-Observe循环，LLM自主决策并观察结果，设置工具调用限制防止过度调用。
- **多Agent集群**: 设置Consultation、Diagnostic、Research和Lead Agent，其中Lead Agent负责任务分解和Agent指派，其余三个Agent并行执行分解后的任务。
- **内置Skills**: 内置知识检索、风险评估、症状模式分析、生活支持建议等 9 个 Skills。
- **多层记忆机制**: 设置短期记忆和长期记忆，短期记忆管理当前会话的对话历史，集成消息压缩机制；长期记忆通过Mem0云服务向量化存储会话总结，支持跨会话相似案例检索。
- **多模态支持**: 支持语音、图像/视频情绪线索和文本输入的融合分析。
- **Agentic RAG**：基于Chroma构建心理健康知识库，通过Agentic RAG提升回答准确性并降低幻觉。
- **Harness约束**：通过YAML文件显式定义Agent能力边界并在运行时自动验证，实现输出自动修复。

## 功能演示
### 登录页面

### 主页面

### 聊天页面

### 视频模式

页面支持学生聊天、管理员报告查看、旁路旧链路回复展示、麦克风录音和多模态输入。模型权重较大，未随仓库提交；模型导入方式见 [LoRA fine-tuning guide](docs/qwen25-7b-lora-finetune-guide.md)。

## 项目流程图

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

## 技术栈

- **后端**: Python, FastAPI, SQLAlchemy, Alembic, Pydantic
- **数据库**: MySQL, Redis，Chroma
- **前端**: HTML/CSS/JavaScript
- **多模态**: 音频使用Whisper, 视频使用POSTER或UniFER
- **部署**: Docker

## 快速开始

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

如果没有 MySQL 或本地模型，只想测试网页和接口：

```bash
DB_URL="sqlite+aiosqlite:///./data/python-dev.db" \
AI_PROVIDER=mock \
USE_CHROMA=false \
MCP_EXCEL_MODE=local \
MCP_EMAIL_MODE=log \
uv run uvicorn app.main:app --host 127.0.0.1 --port 8080
```

## 项目结构

```text
app/
  agent/                 # Agent相关的代码
  api/                   # 管理API调用
  core/                  # 基础功能
  models/                # 模型相关
  knowledge/             # 知识库
skills/                  # 定义的Skills
```

## 注意

- 本系统只用于校园心理健康支持、科普和工程研究，不替代专业心理咨询、医学诊断或紧急救助。
- 高风险场景应联系学校心理中心、辅导员、可信任的人或当地紧急救援服务。
- MCP/tools 默认不全面放开自动执行，避免 Agent 自主触发高副作用操作。
- 本仓库不包含本地大模型权重、运行数据库、Excel 报告和 `.env`。
