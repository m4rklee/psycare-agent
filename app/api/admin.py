from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.core.deps import get_knowledge_service
from app.core.security import require_admin
from app.models.entities import UserAccount
from app.schemas.api import KnowledgeIngestRequest, KnowledgeIngestResponse
from app.services.knowledge import KnowledgeService
from app.services.reports import ReportService

router = APIRouter(prefix="/api", tags=["admin"])
report_service = ReportService()
MAX_KNOWLEDGE_UPLOAD_BYTES = 10 * 1024 * 1024


@router.get("/admin/reports")
async def latest_reports(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_session)],
):
    return await report_service.latest_reports(db)


@router.get("/admin/excel-records")
async def excel_records(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_session)],
):
    return await report_service.excel_records(db)


@router.get("/admin/alerts")
async def alert_records(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_session)],
):
    return await report_service.alert_records(db)


@router.get("/admin/conversations/{session_id}")
async def conversation(
    session_id: str,
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_session)],
):
    response = await report_service.conversation(db, session_id)
    if response is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return response


@router.get("/reports/me")
async def my_reports(
    user: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_session)],
):
    return await report_service.my_reports(db, user.id)


@router.post("/admin/knowledge", response_model=KnowledgeIngestResponse)
async def ingest_knowledge(
    request: KnowledgeIngestRequest,
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_session)],
    knowledge: Annotated[KnowledgeService, Depends(get_knowledge_service)],
) -> KnowledgeIngestResponse:
    count = await knowledge.ingest(db, request.source, request.content)
    return KnowledgeIngestResponse(source=request.source, chunks=count)


@router.post("/admin/knowledge/file", response_model=KnowledgeIngestResponse)
async def ingest_knowledge_file(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_session)],
    knowledge: Annotated[KnowledgeService, Depends(get_knowledge_service)],
    file: UploadFile = File(...),
) -> KnowledgeIngestResponse:
    try:
        source, content = await knowledge.read_upload(
            file.filename or "uploaded-knowledge",
            await read_limited_upload(file),
        )
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文本文件必须使用 UTF-8 编码") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    count = await knowledge.ingest(db, source, content)
    return KnowledgeIngestResponse(source=source, chunks=count)


async def read_limited_upload(file: UploadFile) -> bytes:
    content = await file.read(MAX_KNOWLEDGE_UPLOAD_BYTES + 1)
    if len(content) > MAX_KNOWLEDGE_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件不能超过 10MB")
    return content
