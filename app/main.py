from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import admin, chat, mcp, profile, status
from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.core.deps import get_knowledge_service
from app.core.migrations import run_migrations
from app.services.seed import seed_initial_data


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_migrations()
    async with AsyncSessionLocal() as session:
        await seed_initial_data(session, get_knowledge_service())
    yield


app = FastAPI(title="multimodalAgent Python Backend", version="0.1.0", lifespan=lifespan)
settings = get_settings()

app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(profile.router)
app.include_router(status.router)
app.include_router(mcp.router)

app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if exc.detail is not None else exc.status_code
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": str(detail)},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    messages = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", []) if part not in {"body", "query", "path"})
        message = str(error.get("msg", "validation error"))
        messages.append(f"{loc} {message}".strip())
    return JSONResponse(status_code=400, content={"message": "; ".join(messages)})


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"message": str(exc)})


@app.exception_handler(Exception)
async def unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"message": str(exc)})


@app.get("/actuator/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "UP"}


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.get("/{asset_name}", include_in_schema=False)
async def root_asset(asset_name: str) -> FileResponse:
    allowed = {"app.js", "styles.css", "favicon.svg"}
    if asset_name not in allowed:
        return FileResponse(settings.static_dir / "index.html")
    return FileResponse(settings.static_dir / asset_name)
