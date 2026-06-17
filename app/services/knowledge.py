import json
import math
import re
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from pypdf import PdfReader
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.entities import KnowledgeChunk
from app.services.ai import AiClient, AiMessage
from app.services.prompts import rag_plan_prompt, rag_review_prompt


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int | None
    source: str
    content: str
    score: float


@dataclass(frozen=True)
class AgenticRagResult:
    plan_reason: str
    queries: list[str]
    evidence: list[SearchResult]
    review_reason: str
    sufficient: bool

    @classmethod
    def empty(cls) -> "AgenticRagResult":
        return cls("", [], [], "", True)

    def context_block(self) -> str:
        if not self.evidence:
            return "Agentic RAG：未检索到可用知识。"
        evidence_text = "\n".join(
            f"- [{item.source}] {item.content}" for item in self.evidence
        )
        return (
            f"Agentic RAG 计划：{self.plan_reason}\n"
            f"检索查询：{', '.join(self.queries)}\n"
            f"证据复核：{self.review_reason}；sufficient={self.sufficient}\n"
            f"检索知识：\n{evidence_text}"
        )


def chunk_text(content: str, chunk_size: int, overlap: int) -> list[str]:
    text = content.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks: list[str] = []
    safe_size = max(120, chunk_size)
    safe_overlap = max(0, min(overlap, safe_size // 2))
    index = 0
    while index < len(text):
        end = min(len(text), index + safe_size)
        if end < len(text):
            candidates = [text.rfind(mark, index, end) for mark in ("\n", "。", ".", "?")]
            boundary = max(candidates)
            if boundary > index + safe_size // 2:
                end = boundary + 1
        chunks.append(text[index:end].strip())
        if end >= len(text):
            break
        index = max(0, end - safe_overlap)
    return [chunk for chunk in chunks if chunk]


def vectorize(text: str) -> dict[str, float]:
    normalized = re.sub(r"\s+", " ", text.lower())
    vector: dict[str, float] = {}
    for token in re.split(r"[^\u4e00-\u9fffa-z0-9]+", normalized):
        if token:
            vector[token] = vector.get(token, 0.0) + 1.0
    for left, right in zip(normalized, normalized[1:]):
        if "\u4e00" <= left <= "\u9fff" and "\u4e00" <= right <= "\u9fff":
            token = left + right
            vector[token] = vector.get(token, 0.0) + 1.0
    return vector


def cosine(left: str, right: str) -> float:
    a = vectorize(left)
    b = vectorize(right)
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b)


def cosine_vectors(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def keyword_score(query: str, content: str) -> float:
    terms = [term for term in re.split(r"[\s，。！？、；：,.!?;:]+", query.lower()) if len(term) >= 2]
    if not terms:
        return 0.0
    normalized = content.lower()
    matched = sum(1 for term in terms if term in normalized)
    return min(1.0, matched / len(terms))


class KnowledgeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._chroma_collection_ids: dict[str, str] = {}

    async def ingest(self, session: AsyncSession, source: str, content: str) -> int:
        chunks = chunk_text(content, self.settings.knowledge_chunk_size, self.settings.knowledge_chunk_overlap)
        await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source == source))
        await self._delete_chroma_source(source)
        saved_chunks: list[KnowledgeChunk] = []
        for index, chunk in enumerate(chunks):
            item = KnowledgeChunk(
                source=source,
                source_index=index,
                content=chunk,
                embedding_json=self._serialize_embedding(await self._safe_embedding(chunk)),
            )
            session.add(item)
            saved_chunks.append(item)
        await session.flush()
        await session.commit()
        if self.settings.use_chroma and saved_chunks:
            await self._upsert_chroma(saved_chunks)
        return len(chunks)

    async def ingest_classpath_if_empty(self, session: AsyncSession) -> None:
        count = await session.scalar(select(func.count(KnowledgeChunk.id)))
        if count:
            return
        for path in sorted(self.settings.knowledge_dir.glob("*.md")):
            await self.ingest(session, path.name, path.read_text(encoding="utf-8"))

    async def sync_chroma_from_db(self, session: AsyncSession) -> None:
        if not self.settings.use_chroma:
            return
        chunks = (
            await session.scalars(
                select(KnowledgeChunk).order_by(KnowledgeChunk.source.asc(), KnowledgeChunk.source_index.asc())
            )
        ).all()
        if chunks:
            await self._upsert_chroma(list(chunks))

    async def retrieve(self, session: AsyncSession, query: str, top_k: int) -> list[SearchResult]:
        if self.settings.use_chroma:
            chroma_results = await self._query_chroma(query, top_k)
            if chroma_results:
                return await self._expand_best_context(session, chroma_results, top_k)
        embedding_results = await self._retrieve_by_embedding(session, query, top_k)
        if embedding_results:
            return await self._expand_best_context(session, embedding_results, top_k)
        rows = (await session.scalars(select(KnowledgeChunk))).all()
        ranked = sorted(
            [
                SearchResult(row.id, row.source, row.content, self._hybrid_score(query, row.content))
                for row in rows
            ],
            key=lambda item: item.score,
            reverse=True,
        )
        ranked = [item for item in ranked if item.score > 0.0][:top_k]
        return await self._expand_best_context(session, ranked, top_k)

    async def read_upload(self, filename: str, content: bytes) -> tuple[str, str]:
        if not content:
            raise ValueError("文件内容为空")
        if len(content) > 10 * 1024 * 1024:
            raise ValueError("文件不能超过 10MB")
        source = self._sanitize_source(filename)
        lower = source.lower()
        if lower.endswith(".pdf"):
            text = self._extract_pdf(content)
        elif lower.endswith((".md", ".markdown", ".txt")):
            text = content.decode("utf-8", errors="strict")
        else:
            raise ValueError("仅支持 PDF、Markdown 和 txt 文件")
        if not text.strip():
            raise ValueError("没有从文件中解析出可用文本")
        return source, text

    async def _ensure_chroma_collection(self, client: httpx.AsyncClient) -> str:
        collection = self.settings.chroma_collection
        if collection in self._chroma_collection_ids:
            return self._chroma_collection_ids[collection]
        response = await client.post("/api/v1/collections", json={"name": collection})
        collection_id = self._collection_id_from_response(response, collection)
        if collection_id is None:
            response = await client.get("/api/v1/collections")
            response.raise_for_status()
            collection_id = self._collection_id_from_list(response.json(), collection)
        if collection_id is None:
            raise ValueError(f"Chroma collection {collection} was not created")
        self._chroma_collection_ids[collection] = collection_id
        return collection_id

    async def _delete_chroma_source(self, source: str) -> None:
        if not self.settings.use_chroma:
            return
        try:
            async with httpx.AsyncClient(base_url=self.settings.chroma_base_url, timeout=10) as client:
                collection_id = await self._ensure_chroma_collection(client)
                await client.post(
                    f"/api/v1/collections/{collection_id}/delete",
                    json={"where": {"source": source}},
                )
        except Exception:
            pass

    async def _upsert_chroma(self, chunks: list[KnowledgeChunk]) -> None:
        try:
            async with httpx.AsyncClient(base_url=self.settings.chroma_base_url, timeout=10) as client:
                collection_id = await self._ensure_chroma_collection(client)
                body = {
                    "ids": [str(chunk.id) for chunk in chunks],
                    "documents": [chunk.content for chunk in chunks],
                    "metadatas": [
                        {"source": chunk.source, "sourceIndex": chunk.source_index}
                        for chunk in chunks
                    ],
                }
                embeddings = [self._parse_embedding(chunk.embedding_json) for chunk in chunks]
                if embeddings and all(embedding for embedding in embeddings):
                    body["embeddings"] = embeddings
                await client.post(
                    f"/api/v1/collections/{collection_id}/add",
                    json=body,
                )
        except Exception:
            pass

    async def _query_chroma(self, query: str, top_k: int) -> list[SearchResult]:
        query_embedding = await self._safe_embedding(query)
        if not query_embedding:
            return []
        try:
            async with httpx.AsyncClient(base_url=self.settings.chroma_base_url, timeout=10) as client:
                collection_id = await self._ensure_chroma_collection(client)
                response = await client.post(
                    f"/api/v1/collections/{collection_id}/query",
                    json={
                        "query_embeddings": [query_embedding],
                        "n_results": top_k,
                        "include": ["documents", "metadatas", "distances"],
                    },
                )
                response.raise_for_status()
                data = response.json()
        except Exception:
            return []
        documents = (data.get("documents") or [[]])[0]
        metadatas = (data.get("metadatas") or [[]])[0]
        distances = (data.get("distances") or [[]])[0]
        results: list[SearchResult] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else 1.0
            ids = (data.get("ids") or [[]])[0]
            chunk_id = self._parse_int(ids[index]) if index < len(ids) else None
            results.append(SearchResult(chunk_id, metadata.get("source", "chroma"), document, 1.0 - float(distance)))
        return results

    async def _retrieve_by_embedding(self, session: AsyncSession, query: str, top_k: int) -> list[SearchResult]:
        query_embedding = await self._safe_embedding(query)
        if not query_embedding:
            return []
        rows = (await session.scalars(select(KnowledgeChunk))).all()
        results = [
            SearchResult(row.id, row.source, row.content, cosine_vectors(query_embedding, self._parse_embedding(row.embedding_json)))
            for row in rows
        ]
        return sorted([item for item in results if item.score > 0.0], key=lambda item: item.score, reverse=True)[:top_k]

    async def _expand_best_context(
        self, session: AsyncSession, ranked: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not ranked:
            return ranked
        best = await self._expand(session, ranked[0])
        results = [best]
        for result in ranked[1:]:
            if result.chunk_id is not None and result.chunk_id == best.chunk_id:
                continue
            results.append(result)
            if len(results) >= top_k:
                break
        return results

    async def _expand(self, session: AsyncSession, result: SearchResult) -> SearchResult:
        if result.chunk_id is None:
            return result
        chunk = await session.get(KnowledgeChunk, result.chunk_id)
        if chunk is None:
            return result
        neighbors = (
            await session.scalars(
                select(KnowledgeChunk)
                .where(
                    KnowledgeChunk.source == chunk.source,
                    KnowledgeChunk.source_index >= max(0, chunk.source_index - 1),
                    KnowledgeChunk.source_index <= chunk.source_index + 1,
                )
                .order_by(KnowledgeChunk.source_index.asc())
            )
        ).all()
        return SearchResult(chunk.id, chunk.source, "\n\n".join(item.content for item in neighbors), result.score)

    async def _safe_embedding(self, text: str) -> list[float]:
        if not self.settings.openai_api_key:
            return []
        try:
            client = AsyncOpenAI(api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url)
            response = await client.embeddings.create(model=self.settings.openai_embedding_model, input=text)
            return list(response.data[0].embedding)
        except Exception:
            return []

    def _serialize_embedding(self, embedding: list[float]) -> str | None:
        return json.dumps(embedding) if embedding else None

    def _parse_embedding(self, value: str | None) -> list[float]:
        if not value:
            return []
        try:
            data = json.loads(value)
            return [float(item) for item in data]
        except Exception:
            return []

    def _hybrid_score(self, query: str, content: str) -> float:
        return cosine(query, content) * 0.75 + keyword_score(query, content) * 0.25

    def _sanitize_source(self, filename: str) -> str:
        source = (filename or "uploaded-knowledge").strip() or "uploaded-knowledge"
        source = re.sub(r"[\\/]+", "-", source)
        return source[-180:]

    def _extract_pdf(self, content: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ValueError(f"PDF 文本解析失败：{exc}") from exc

    def _parse_int(self, value: object) -> int | None:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    def _collection_id_from_response(self, response: httpx.Response, collection: str) -> str | None:
        if response.status_code in {200, 201}:
            try:
                data = response.json()
            except Exception:
                return None
            if isinstance(data, dict):
                return str(data.get("id") or "") or None
        if response.status_code != 409:
            response.raise_for_status()
        return None

    def _collection_id_from_list(self, data: object, collection: str) -> str | None:
        if not isinstance(data, list):
            return None
        for item in data:
            if isinstance(item, dict) and item.get("name") == collection and item.get("id"):
                return str(item["id"])
        return None


class AgenticRagService:
    def __init__(self, settings: Settings, ai_client: AiClient, knowledge_service: KnowledgeService) -> None:
        self.settings = settings
        self.ai_client = ai_client
        self.knowledge_service = knowledge_service

    async def retrieve(
        self, session: AsyncSession, user_input: str, history: list[AiMessage]
    ) -> AgenticRagResult:
        reason, queries = await self._plan(user_input, history)
        evidence = await self._search(session, queries, self.settings.rag_top_k)
        sufficient, review_reason, follow_up = await self._review(user_input, evidence)
        if not sufficient:
            evidence = self._dedupe(
                evidence + await self._search(session, follow_up, self.settings.rag_top_k),
                self.settings.rag_top_k,
            )
            sufficient, review_reason, _ = await self._review(user_input, evidence)
        return AgenticRagResult(reason, queries, evidence, review_reason, sufficient)

    async def _plan(self, user_input: str, history: list[AiMessage]) -> tuple[str, list[str]]:
        try:
            raw = await self.ai_client.complete(rag_plan_prompt(history, user_input))
            data = self._extract_json(raw)
            queries = [str(item).strip() for item in data.get("queries", []) if str(item).strip()]
            return data.get("reason", "围绕用户当前心理支持需求检索校园心理健康知识。"), (queries or [user_input])[:3]
        except Exception:
            return "模型规划失败，使用用户原问题直接检索。", [user_input]

    async def _review(self, user_input: str, evidence: list[SearchResult]) -> tuple[bool, str, list[str]]:
        evidence_text = "\n\n".join(f"- [{item.source}] {item.content}" for item in evidence)
        try:
            raw = await self.ai_client.complete(rag_review_prompt(user_input, evidence_text))
            data = self._extract_json(raw)
            follow_up = [str(item).strip() for item in data.get("followUpQueries", []) if str(item).strip()]
            return bool(data.get("sufficient", False)), data.get("reason", "证据覆盖度不足。"), follow_up
        except Exception:
            return bool(evidence), "已找到可用知识片段。" if evidence else "未找到可用证据。", [user_input]

    async def _search(self, session: AsyncSession, queries: list[str], top_k: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        for query in queries:
            if query:
                results.extend(await self.knowledge_service.retrieve(session, query, top_k))
        return self._dedupe(results, top_k)

    def _dedupe(self, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        best: dict[str, SearchResult] = {}
        for result in results:
            key = f"id:{result.chunk_id}" if result.chunk_id is not None else f"{result.source}:{result.content}"
            if key not in best or result.score > best[key].score:
                best[key] = result
        return sorted(best.values(), key=lambda item: item.score, reverse=True)[:top_k]

    def _extract_json(self, raw: str) -> dict:
        start = raw.find("{")
        end = raw.rfind("}")
        text = raw[start : end + 1] if start >= 0 and end > start else raw
        return json.loads(text)
