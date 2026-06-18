from app.core.config import Settings


def test_mcp_urls_default_to_server_port() -> None:
    settings = Settings(ai_provider="mock", server_port=8099, mcp_excel_url="", mcp_email_url="")
    assert settings.mcp_excel_url == "http://127.0.0.1:8099/mcp"
    assert settings.mcp_email_url == "http://127.0.0.1:8099/mcp"
