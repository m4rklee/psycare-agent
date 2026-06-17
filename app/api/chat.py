import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.core.deps import (
    get_agentic_rag_service,
    get_agent_runtime,
    get_ai_client,
    get_assessment_service,
    get_intent_classifier,
    get_multimodal_service,
    get_short_memory,
    get_tool_service,
    settings,
)
from app.core.security import require_student
from app.models.entities import UserAccount
from app.schemas.api import ChatRequest, ChatSessionSummaryResponse, ConversationResponse
from app.services.chat import ChatService
from app.services.multimodal import (
    BROWSER_RECORDING_MOCK_ERROR_MESSAGE,
    POSTER_PP_ERROR_MESSAGE,
    TRANSCRIPTION_ERROR_MESSAGE,
    MultimodalInputService,
    PosterPPAnalysisError,
    WhisperTranscriptionError,
)
from app.services.reports import ReportService

router = APIRouter(prefix="/api/chat", tags=["chat"])
report_service = ReportService()


def sse_error_response(content: str, status_code: int = 200) -> StreamingResponse:
    async def error_stream():
        payload = json.dumps({"type": "error", "content": content}, ensure_ascii=False, separators=(",", ":"))
        yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=status_code)


def transcription_error_message(error: WhisperTranscriptionError) -> str:
    message = str(error)
    if message == BROWSER_RECORDING_MOCK_ERROR_MESSAGE:
        return message
    return TRANSCRIPTION_ERROR_MESSAGE


def chat_service() -> ChatService:
    return ChatService(
        get_ai_client(),
        get_intent_classifier(),
        get_assessment_service(),
        get_agentic_rag_service(),
        get_tool_service(),
        get_short_memory(),
        settings(),
        get_agent_runtime(),
    )


@router.get("/sessions", response_model=list[ChatSessionSummaryResponse])
async def chat_sessions(
    user: Annotated[UserAccount, Depends(require_student)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> list[ChatSessionSummaryResponse]:
    return await report_service.student_sessions(db, user.id)


@router.get("/sessions/{session_id}", response_model=ConversationResponse)
async def chat_session(
    session_id: str,
    user: Annotated[UserAccount, Depends(require_student)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> ConversationResponse:
    response = await report_service.student_conversation(db, user.id, session_id)
    if response is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return response


@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    user: Annotated[UserAccount, Depends(require_student)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> StreamingResponse:
    return StreamingResponse(
        chat_service().stream_chat(db, user, request),
        media_type="text/event-stream",
    )


@router.post("/multimodal/stream")
async def stream_multimodal(
    user: Annotated[UserAccount, Depends(require_student)],
    db: Annotated[AsyncSession, Depends(get_session)],
    multimodal_service: Annotated[MultimodalInputService, Depends(get_multimodal_service)],
    sessionId: str | None = Form(default=None),
    message: str | None = Form(default=None),
    audio: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
    video: UploadFile | None = File(default=None),
) -> StreamingResponse:
    has_text = bool(message and message.strip())
    if not has_text and not any((audio, image, video)):
        return sse_error_response("请至少输入文字或上传一个多模态文件。", status_code=400)
    text = message.strip() if has_text and message else "学生上传了多模态内容，希望获得支持。"
    try:
        analysis = await multimodal_service.analyze(text, audio, image, video)
    except WhisperTranscriptionError as exc:
        return sse_error_response(transcription_error_message(exc))
    return StreamingResponse(
        chat_service().stream_chat(db, user, ChatRequest(sessionId=sessionId, message=text), analysis),
        media_type="text/event-stream",
    )


@router.post("/video/stream")
async def stream_video_chat(
    user: Annotated[UserAccount, Depends(require_student)],
    db: Annotated[AsyncSession, Depends(get_session)],
    multimodal_service: Annotated[MultimodalInputService, Depends(get_multimodal_service)],
    audio: UploadFile = File(...),
    frame: UploadFile = File(...),
    sessionId: str | None = Form(default=None),
    message: str | None = Form(default=None),
    preprocessMode: str | None = Form(default=None),
    cropBox: str | None = Form(default=None),
    sourceWidth: int | None = Form(default=None),
    sourceHeight: int | None = Form(default=None),
    outputSize: int | None = Form(default=None),
    fallback: str | None = Form(default=None),
) -> StreamingResponse:
    text = message.strip() if message and message.strip() else "学生正在进行视频心理支持对话。"
    try:
        analysis = await multimodal_service.analyze_video_chat(
            text,
            audio,
            frame,
            {
                "preprocessMode": preprocessMode,
                "cropBox": cropBox,
                "sourceWidth": sourceWidth,
                "sourceHeight": sourceHeight,
                "outputSize": outputSize,
                "fallback": fallback,
            },
        )
    except WhisperTranscriptionError as exc:
        return sse_error_response(transcription_error_message(exc))
    except PosterPPAnalysisError:
        return sse_error_response(POSTER_PP_ERROR_MESSAGE)
    return StreamingResponse(
        chat_service().stream_chat(db, user, ChatRequest(sessionId=sessionId, message=text), analysis),
        media_type="text/event-stream",
    )
