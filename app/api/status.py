from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.config import Settings
from app.core.deps import settings
from app.core.security import current_user
from app.models.entities import UserAccount
from app.schemas.api import AgentStatusResponse

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.get("/status", response_model=AgentStatusResponse)
async def status(
    _: Annotated[UserAccount, Depends(current_user)],
    cfg: Annotated[Settings, Depends(settings)],
) -> AgentStatusResponse:
    model = cfg.ollama_model if cfg.ai_provider == "ollama" else cfg.openai_model
    real_model_enabled = cfg.ai_provider in {"ollama", "openai"}
    return AgentStatusResponse(
        provider=cfg.ai_provider,
        model=model,
        realModelEnabled=real_model_enabled,
        chromaEnabled=cfg.use_chroma,
        ragTopK=cfg.rag_top_k,
        note="正在使用真实大模型客户端。" if real_model_enabled else "当前为本地 mock 演示模式，不会调用大模型。",
        knowledgeMode="chroma" if cfg.use_chroma else "local",
        mcpExcelMode=cfg.mcp_excel_mode,
        mcpEmailMode=cfg.mcp_email_mode,
    )
