import pytest
from fastapi import HTTPException

from app.api.admin import MAX_KNOWLEDGE_UPLOAD_BYTES, read_limited_upload
from app.core.config import Settings
from app.models.entities import KnowledgeChunk
from app.services.seed import seed_initial_data
from app.services.knowledge import KnowledgeService, chunk_text, cosine


def test_chunk_text_respects_natural_boundaries() -> None:
    text = "第一段说明心理支持。\n第二段说明校园资源。第三段说明风险处理。"
    chunks = chunk_text(text, chunk_size=120, overlap=10)
    assert chunks == [text]


def test_cosine_gives_related_chinese_text_positive_score() -> None:
    score = cosine("焦虑 失眠 压力", "压力很大导致焦虑和失眠")
    assert score > 0


@pytest.mark.asyncio
async def test_read_upload_accepts_utf8_text() -> None:
    service = KnowledgeService(Settings(ai_provider="mock"))
    source, text = await service.read_upload("../risk-policy.txt", "危机干预".encode())
    assert source == "..-risk-policy.txt"
    assert text == "危机干预"


@pytest.mark.asyncio
async def test_read_upload_rejects_unsupported_file_type() -> None:
    service = KnowledgeService(Settings(ai_provider="mock"))
    with pytest.raises(ValueError, match="仅支持"):
        await service.read_upload("notes.docx", b"hello")


@pytest.mark.asyncio
async def test_read_upload_rejects_large_file() -> None:
    service = KnowledgeService(Settings(ai_provider="mock"))
    with pytest.raises(ValueError, match="10MB"):
        await service.read_upload("large.txt", b"x" * (10 * 1024 * 1024 + 1))


class FakeUpload:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.content[:size]


@pytest.mark.asyncio
async def test_limited_upload_reads_at_most_limit_plus_one() -> None:
    upload = FakeUpload(b"x" * (MAX_KNOWLEDGE_UPLOAD_BYTES + 100))

    with pytest.raises(HTTPException) as exc_info:
        await read_limited_upload(upload)  # type: ignore[arg-type]

    assert upload.read_sizes == [MAX_KNOWLEDGE_UPLOAD_BYTES + 1]
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "文件不能超过 10MB"


@pytest.mark.asyncio
async def test_limited_upload_accepts_file_at_limit() -> None:
    upload = FakeUpload(b"x" * MAX_KNOWLEDGE_UPLOAD_BYTES)

    content = await read_limited_upload(upload)  # type: ignore[arg-type]

    assert len(content) == MAX_KNOWLEDGE_UPLOAD_BYTES
    assert upload.read_sizes == [MAX_KNOWLEDGE_UPLOAD_BYTES + 1]


class EmptyKnowledgeSession:
    async def scalar(self, statement):
        return 0


@pytest.mark.asyncio
async def test_classpath_ingest_uses_full_filename_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "campus-policy.md").write_text("校园心理支持", encoding="utf-8")
    service = KnowledgeService(Settings(ai_provider="mock"))
    monkeypatch.setattr(Settings, "knowledge_dir", property(lambda self: knowledge_dir))
    calls: list[tuple[str, str]] = []

    async def fake_ingest(session, source: str, content: str) -> int:
        calls.append((source, content))
        return 1

    service.ingest = fake_ingest  # type: ignore[method-assign]
    await service.ingest_classpath_if_empty(EmptyKnowledgeSession())  # type: ignore[arg-type]
    assert calls == [("campus-policy.md", "校园心理支持")]


class ExistingKnowledgeSession:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = chunks
        self.commits = 0

    async def scalar(self, statement):
        return 1

    async def scalars(self, statement):
        return ExistingKnowledgeRows(self.chunks)


class ExistingKnowledgeRows:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = chunks

    def all(self) -> list[KnowledgeChunk]:
        return self.chunks


@pytest.mark.asyncio
async def test_seed_syncs_existing_db_chunks_to_chroma() -> None:
    chunk = KnowledgeChunk(id=7, source="campus.md", source_index=0, content="校园心理支持")
    service = KnowledgeService(Settings(ai_provider="mock", use_chroma=True))
    synced: list[list[KnowledgeChunk]] = []

    async def fake_upsert_chroma(chunks: list[KnowledgeChunk]) -> None:
        synced.append(chunks)

    service._upsert_chroma = fake_upsert_chroma  # type: ignore[method-assign]

    await seed_initial_data(ExistingKnowledgeSession([chunk]), service)  # type: ignore[arg-type]

    assert synced == [[chunk]]


class ChromaClient:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []

    async def post(self, path: str, json: dict):
        self.posts.append((path, json))
        return ChromaResponse(200, {"id": "collection-uuid", "name": json["name"]})

    async def get(self, path: str):
        self.gets.append(path)
        return ChromaResponse(200, [{"id": "collection-uuid", "name": "test_collection"}])


class ExistingChromaClient(ChromaClient):
    async def post(self, path: str, json: dict):
        self.posts.append((path, json))
        return ChromaResponse(409, {"error": "UniqueConstraintError"})


class ChromaResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.payload = payload

    def json(self) -> object:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.mark.asyncio
async def test_chroma_collection_is_ensured_once() -> None:
    service = KnowledgeService(Settings(ai_provider="mock", chroma_collection="test_collection"))
    client = ChromaClient()
    assert await service._ensure_chroma_collection(client) == "collection-uuid"  # type: ignore[arg-type]
    assert await service._ensure_chroma_collection(client) == "collection-uuid"  # type: ignore[arg-type]
    assert client.posts == [("/api/v1/collections", {"name": "test_collection"})]
    assert client.gets == []


@pytest.mark.asyncio
async def test_chroma_collection_id_is_loaded_when_collection_already_exists() -> None:
    service = KnowledgeService(Settings(ai_provider="mock", chroma_collection="test_collection"))
    client = ExistingChromaClient()

    assert await service._ensure_chroma_collection(client) == "collection-uuid"  # type: ignore[arg-type]

    assert client.posts == [("/api/v1/collections", {"name": "test_collection"})]
    assert client.gets == ["/api/v1/collections"]
