from app.services.ai import AiMessage
from app.services.memory import ShortTermMemoryService
from app.services.rules import sanitize


class ShortTermMemoryAdapter:
    def __init__(self, short_memory: ShortTermMemoryService) -> None:
        self.short_memory = short_memory

    async def load(self, public_session_id: str) -> list[AiMessage]:
        messages = await self.short_memory.recent(public_session_id)
        return [AiMessage(message.role.value.lower(), sanitize(message.content)) for message in messages]
