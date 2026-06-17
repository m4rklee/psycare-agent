import asyncio

from alembic import command
from alembic.config import Config

from app.core.config import Settings, get_settings


def alembic_config(settings: Settings) -> Config:
    config = Config(str(settings.project_root / "alembic.ini"))
    config.set_main_option("script_location", str(settings.project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", settings.db_url)
    return config


def _upgrade_to_head(settings: Settings) -> None:
    command.upgrade(alembic_config(settings), "head")


async def run_migrations(settings: Settings | None = None) -> None:
    await asyncio.to_thread(_upgrade_to_head, settings or get_settings())
