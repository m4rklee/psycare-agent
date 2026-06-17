from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.entities import AlertRecord, ChatMessage, ChatSession, PsychologicalReport, UserAccount
from app.models.enums import MessageRole, ToolStatus
from app.schemas.api import (
    AlertRecordResponse,
    ChatSessionSummaryResponse,
    ConversationMessageResponse,
    ConversationResponse,
    ExcelRecordResponse,
    ReportResponse,
)


def _roles(user: UserAccount) -> set[str]:
    return {role.role for role in user.roles}


def _is_student_user(user: UserAccount) -> bool:
    return "ROLE_ADMIN" not in _roles(user)


def report_response(report: PsychologicalReport) -> ReportResponse:
    return ReportResponse(
        id=report.id,
        userId=report.user.id,
        username=report.user.username,
        sessionId=report.session.public_id if report.session else None,
        intent=report.intent,
        emotion=report.emotion,
        emotionScore=report.emotion_score,
        riskLevel=report.risk_level,
        confidence=report.confidence,
        summary=report.summary,
        emotionTags=report.emotion_tags,
        excelStatus=report.excel_status,
        emailStatus=report.email_status,
        createdAt=report.created_at,
    )


def excel_response(report: PsychologicalReport) -> ExcelRecordResponse:
    return ExcelRecordResponse(
        reportId=report.id,
        userId=report.user.id,
        username=report.user.username,
        sessionId=report.session.public_id if report.session else None,
        intent=report.intent,
        emotion=report.emotion,
        emotionScore=report.emotion_score,
        riskLevel=report.risk_level,
        confidence=report.confidence,
        summary=report.summary,
        emotionTags=report.emotion_tags,
        content=report.content,
        excelStatus=report.excel_status,
        createdAt=report.created_at,
    )


def alert_response(alert: AlertRecord) -> AlertRecordResponse:
    report = alert.report
    return AlertRecordResponse(
        id=alert.id,
        reportId=report.id,
        userId=report.user.id,
        username=report.user.username,
        sessionId=report.session.public_id if report.session else None,
        riskLevel=report.risk_level,
        summary=report.summary,
        recipient=alert.recipient,
        status=alert.status,
        errorMessage=alert.error_message,
        attempts=alert.attempts,
        createdAt=alert.created_at,
        updatedAt=alert.updated_at,
    )


def chat_session_summary(session: ChatSession) -> ChatSessionSummaryResponse:
    return ChatSessionSummaryResponse(
        sessionId=session.public_id,
        title=session.title,
        createdAt=session.created_at,
        updatedAt=session.updated_at,
    )


def conversation_response(chat_session: ChatSession, messages: list[ChatMessage]) -> ConversationResponse:
    return ConversationResponse(
        sessionId=chat_session.public_id,
        title=chat_session.title,
        userId=chat_session.user.id,
        username=chat_session.user.username,
        displayName=chat_session.user.display_name,
        createdAt=chat_session.created_at,
        updatedAt=chat_session.updated_at,
        messages=[
            ConversationMessageResponse(
                id=message.id,
                role=message.role,
                content=message.content,
                createdAt=message.created_at,
            )
            for message in messages
        ],
    )


class ReportService:
    async def student_sessions(self, session: AsyncSession, user_id: int) -> list[ChatSessionSummaryResponse]:
        rows = (
            await session.scalars(
                select(ChatSession)
                .where(ChatSession.user_id == user_id)
                .order_by(ChatSession.updated_at.desc())
                .limit(50)
            )
        ).all()
        return [chat_session_summary(row) for row in rows]

    async def student_conversation(
        self, session: AsyncSession, user_id: int, session_id: str
    ) -> ConversationResponse | None:
        chat_session = await session.scalar(
            select(ChatSession)
            .options(selectinload(ChatSession.user))
            .where(ChatSession.public_id == session_id, ChatSession.user_id == user_id)
        )
        if not chat_session:
            return None
        messages = await self._conversation_messages(session, chat_session.id)
        return conversation_response(chat_session, messages)

    async def my_reports(self, session: AsyncSession, user_id: int) -> list[ReportResponse]:
        reports = (
            await session.scalars(
                select(PsychologicalReport)
                .options(selectinload(PsychologicalReport.user), selectinload(PsychologicalReport.session))
                .where(PsychologicalReport.user_id == user_id)
                .order_by(PsychologicalReport.created_at.desc())
                .limit(50)
            )
        ).all()
        return [report_response(report) for report in reports]

    async def latest_reports(self, session: AsyncSession) -> list[ReportResponse]:
        reports = (
            await session.scalars(
                select(PsychologicalReport)
                .options(
                    selectinload(PsychologicalReport.user).selectinload(UserAccount.roles),
                    selectinload(PsychologicalReport.session),
                )
                .order_by(PsychologicalReport.created_at.desc())
                .limit(100)
            )
        ).all()
        return [report_response(report) for report in reports if _is_student_user(report.user)]

    async def excel_records(self, session: AsyncSession) -> list[ExcelRecordResponse]:
        reports = (
            await session.scalars(
                select(PsychologicalReport)
                .options(
                    selectinload(PsychologicalReport.user).selectinload(UserAccount.roles),
                    selectinload(PsychologicalReport.session),
                )
                .where(PsychologicalReport.excel_status == ToolStatus.SUCCESS)
                .order_by(PsychologicalReport.created_at.desc())
                .limit(100)
            )
        ).all()
        return [excel_response(report) for report in reports if _is_student_user(report.user)]

    async def alert_records(self, session: AsyncSession) -> list[AlertRecordResponse]:
        alerts = (
            await session.scalars(
                select(AlertRecord)
                .options(
                    selectinload(AlertRecord.report)
                    .selectinload(PsychologicalReport.user)
                    .selectinload(UserAccount.roles),
                    selectinload(AlertRecord.report).selectinload(PsychologicalReport.session),
                )
                .order_by(AlertRecord.created_at.desc())
                .limit(100)
            )
        ).all()
        return [alert_response(alert) for alert in alerts if _is_student_user(alert.report.user)]

    async def conversation(self, session: AsyncSession, session_id: str) -> ConversationResponse | None:
        chat_session = await session.scalar(
            select(ChatSession)
            .options(selectinload(ChatSession.user).selectinload(UserAccount.roles))
            .where(ChatSession.public_id == session_id)
        )
        if not chat_session or not _is_student_user(chat_session.user):
            return None
        messages = await self._conversation_messages(session, chat_session.id)
        return conversation_response(chat_session, messages)

    async def _conversation_messages(self, session: AsyncSession, chat_session_id: int) -> list[ChatMessage]:
        return (
            await session.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == chat_session_id, ChatMessage.role != MessageRole.SYSTEM)
                .order_by(ChatMessage.created_at.asc())
            )
        ).all()
