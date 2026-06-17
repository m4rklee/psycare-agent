import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import ValidationError

import app.main as main_module
from app.api.profile import profile
from app.core.security import require_admin
from app.main import (
    app,
    health,
    http_exception_handler,
    unexpected_exception_handler,
    validation_exception_handler,
    value_error_handler,
)
from app.schemas.api import ChatRequest, KnowledgeIngestRequest


class FakeRole:
    def __init__(self, role: str) -> None:
        self.role = role


class FakeUser:
    def __init__(self, roles: list[str]) -> None:
        self.id = 1
        self.username = "admin" if "ROLE_ADMIN" in roles else "student"
        self.display_name = "Counselor Admin" if "ROLE_ADMIN" in roles else "student"
        self.roles = [FakeRole(role) for role in roles]


def test_expected_routes_are_registered() -> None:
    routes = {getattr(route, "path", "") for route in app.routes}
    assert "/api/chat/sessions" in routes
    assert "/api/chat/sessions/{session_id}" in routes
    assert "/api/chat/stream" in routes
    assert "/api/chat/multimodal/stream" in routes
    assert "/api/chat/video/stream" in routes
    assert "/api/admin/reports" in routes
    assert "/api/admin/knowledge" in routes
    assert "/mcp" in routes
    assert "/actuator/health" in routes


@pytest.mark.asyncio
async def test_health_matches_spring_actuator_shape() -> None:
    assert await health() == {"status": "UP"}


@pytest.mark.asyncio
async def test_lifespan_runs_migrations_before_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    async def fake_run_migrations() -> None:
        calls.append("migrate")

    class FakeSession:
        async def __aenter__(self) -> str:
            calls.append("session_enter")
            return "db-session"

        async def __aexit__(self, exc_type, exc, tb) -> None:
            calls.append("session_exit")

    async def fake_seed_initial_data(session: str, knowledge_service: str) -> None:
        calls.append(("seed", session, knowledge_service))

    monkeypatch.setattr(main_module, "run_migrations", fake_run_migrations)
    monkeypatch.setattr(main_module, "AsyncSessionLocal", lambda: FakeSession())
    monkeypatch.setattr(main_module, "seed_initial_data", fake_seed_initial_data)
    monkeypatch.setattr(main_module, "get_knowledge_service", lambda: "knowledge-service")

    async with main_module.lifespan(main_module.app):
        calls.append("yielded")

    assert calls == [
        "migrate",
        "session_enter",
        ("seed", "db-session", "knowledge-service"),
        "session_exit",
        "yielded",
    ]


@pytest.mark.asyncio
async def test_profile_roles_match_spring_authority_shape() -> None:
    response = await profile(FakeUser(["ROLE_USER", "ROLE_ADMIN"]))  # type: ignore[arg-type]

    payload = response.model_dump()
    assert payload["roles"] == [{"authority": "ROLE_ADMIN"}, {"authority": "ROLE_USER"}]
    assert any(role["authority"] == "ROLE_ADMIN" for role in payload["roles"])


def test_chat_request_rejects_blank_message_like_java_not_blank() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_knowledge_request_rejects_blank_fields_like_java_not_blank() -> None:
    with pytest.raises(ValidationError):
        KnowledgeIngestRequest(source="knowledge.md", content="   ")
    with pytest.raises(ValidationError):
        KnowledgeIngestRequest(source="   ", content="校园心理支持")


@pytest.mark.asyncio
async def test_http_exception_uses_api_message_shape() -> None:
    response = await http_exception_handler(
        None,  # type: ignore[arg-type]
        HTTPException(status_code=403, detail="Admin role required", headers={"X-Test": "yes"}),
    )
    assert response.status_code == 403
    assert response.headers["x-test"] == "yes"
    assert json.loads(response.body) == {"message": "Admin role required"}


@pytest.mark.asyncio
async def test_validation_exception_uses_java_bad_request_shape() -> None:
    response = await validation_exception_handler(
        None,  # type: ignore[arg-type]
        RequestValidationError(
            [{"loc": ("body", "message"), "msg": "String should have at least 1 character"}]
        ),
    )
    assert response.status_code == 400
    assert json.loads(response.body) == {
        "message": "message String should have at least 1 character"
    }


@pytest.mark.asyncio
async def test_value_error_uses_api_message_shape() -> None:
    response = await value_error_handler(None, ValueError("文件内容为空"))  # type: ignore[arg-type]
    assert response.status_code == 400
    assert json.loads(response.body) == {"message": "文件内容为空"}


@pytest.mark.asyncio
async def test_unexpected_exception_uses_api_message_shape() -> None:
    response = await unexpected_exception_handler(None, RuntimeError("boom"))  # type: ignore[arg-type]
    assert response.status_code == 500
    assert json.loads(response.body) == {"message": "boom"}


def test_reports_me_route_requires_admin_like_java_security_config() -> None:
    route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/api/reports/me"
    )
    assert require_admin in {dependency.call for dependency in route.dependant.dependencies}


@pytest.mark.asyncio
async def test_require_admin_rejects_non_admin() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await require_admin(FakeUser(["ROLE_USER"]))  # type: ignore[arg-type]
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_admin_accepts_admin() -> None:
    user = FakeUser(["ROLE_USER", "ROLE_ADMIN"])
    assert await require_admin(user) is user  # type: ignore[arg-type]


def test_python_dockerfile_uses_locked_uv_dependencies() -> None:
    dockerfile = Path("Dockerfile.python").read_text()

    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "uv sync --locked --no-dev" in dockerfile
    assert "--resolution=lowest-direct" not in dockerfile


def test_compose_pins_chroma_to_v1_compatible_image() -> None:
    compose = Path("docker-compose.yml").read_text()

    assert "image: chromadb/chroma:0.5.23" in compose
    assert "image: chromadb/chroma:latest" not in compose
