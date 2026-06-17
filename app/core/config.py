from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    server_host: str = "127.0.0.1"
    server_port: int = 8080

    db_url: str = (
        "mysql+aiomysql://multimodalAgent:multimodalAgent@127.0.0.1:3306/"
        "multimodalAgent?charset=utf8mb4"
    )
    redis_url: str = "redis://127.0.0.1:6379/0"

    ai_provider: str = "ollama"
    ai_temperature: float = 0.35
    ai_max_tokens: int = 512
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "multimodalAgent-qwen2.5-7b-ft:latest"
    openai_base_url: str = "https://api.openai.com"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    chat_history_limit: int = 10
    chat_short_memory_ttl_hours: int = 24
    agent_skills_dir: str = "./skills"
    agent_task_timeout_seconds: float = 30.0
    agent_enable_background_orchestration: bool = True
    agent_max_capability_calls: int = 3
    agent_capability_timeout_seconds: float = 8.0
    agent_enable_mcp_auto_call: bool = False

    rag_top_k: int = 4
    use_chroma: bool = True
    chroma_base_url: str = "http://127.0.0.1:8000"
    chroma_collection: str = "multimodalAgent_knowledge"
    knowledge_chunk_size: int = 512
    knowledge_chunk_overlap: int = 64

    multimodal_text_weight: float = 0.1
    multimodal_audio_weight: float = 0.4
    multimodal_visual_weight: float = 0.5
    whisper_mode: str = "mock"
    whisper_base_url: str = "https://api.openai.com"
    whisper_api_key: str = ""
    whisper_model: str = "whisper-1"
    mediapipe_mode: str = Field(default="local-rule", alias="MEDIAPIPE_MODE")
    mediapipe_url: str = Field(default="http://127.0.0.1:8090/analyze", alias="MEDIAPIPE_URL")
    poster_pp_url: str = Field(default="http://127.0.0.1:8096", alias="POSTER_PP_URL")

    mcp_excel_mode: str = "mcp"
    mcp_excel_url: str = ""
    mcp_excel_local_path: str = "./data/multimodalAgent-reports.xlsx"
    mcp_email_mode: str = "mcp"
    mcp_email_url: str = ""
    alert_mail_from: str = "multimodalAgent-alerts@example.com"
    alert_mail_recipients: str = "counselor-alerts@example.com"
    alert_mail_max_retries: int = 2
    mail_host: str = "127.0.0.1"
    mail_port: int = 1025
    mail_username: str = ""
    mail_password: str = ""
    mail_smtp_auth: bool = False
    mail_smtp_starttls_enable: bool = False
    mail_smtp_ssl_enable: bool = False

    @field_validator("ai_provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return value.lower().strip()

    @model_validator(mode="after")
    def fill_mcp_urls(self) -> "Settings":
        default_mcp_url = f"http://127.0.0.1:{self.server_port}/mcp"
        if not self.mcp_excel_url:
            self.mcp_excel_url = default_mcp_url
        if not self.mcp_email_url:
            self.mcp_email_url = default_mcp_url
        return self

    @property
    def alert_recipients(self) -> list[str]:
        return [item.strip() for item in self.alert_mail_recipients.split(",") if item.strip()]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def static_dir(self) -> Path:
        return self.project_root / "app" / "static"

    @property
    def knowledge_dir(self) -> Path:
        return self.project_root / "app" / "knowledge"


@lru_cache
def get_settings() -> Settings:
    return Settings()
