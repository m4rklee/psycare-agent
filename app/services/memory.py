import json
from dataclasses import asdict, dataclass

from redis.asyncio import Redis

from app.core.config import Settings
from app.models.enums import MessageRole


@dataclass(frozen=True)
class MemoryMessage:
    role: MessageRole
    content: str


class ShortTermMemoryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.redis = Redis.from_url(settings.redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"multimodalAgent:chat:{session_id}"

    async def append(self, session_id: str, role: MessageRole, content: str) -> None:
        try:
            item = json.dumps({"role": role.value, "content": content}, ensure_ascii=False)
            key = self._key(session_id)
            await self.redis.rpush(key, item)
            await self.redis.ltrim(key, -40, -1)
            await self.redis.expire(key, self.settings.chat_short_memory_ttl_hours * 3600)
        except Exception:
            pass

    async def recent(self, session_id: str) -> list[MemoryMessage]:
        try:
            values = await self.redis.lrange(self._key(session_id), 0, -1)
        except Exception:
            return []
        messages: list[MemoryMessage] = []
        for value in values:
            try:
                data = json.loads(value)
                messages.append(MemoryMessage(MessageRole(data["role"]), data["content"]))
            except Exception:
                continue
        return messages

    async def refresh(self, session_id: str, messages: list[MemoryMessage]) -> None:
        try:
            key = self._key(session_id)
            await self.redis.delete(key)
            if messages:
                await self.redis.rpush(
                    key,
                    *[
                        json.dumps(asdict(message), default=str, ensure_ascii=False)
                        for message in messages[-40:]
                    ],
                )
                await self.redis.expire(key, self.settings.chat_short_memory_ttl_hours * 3600)
        except Exception:
            pass
