from typing import Any

from fastapi import APIRouter
from app.core.config import get_settings
from app.services.tools import AlertNotifier, ExcelReportWriter

router = APIRouter(tags=["mcp"])


@router.post("/mcp")
async def mcp(payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("id")
    try:
        method = payload.get("method", "")
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "multimodalAgent-mcp-server", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "multimodalAgent.excel.write_report",
                        "description": "Write a psychological report row to the multimodalAgent Excel ledger.",
                        "inputSchema": {
                            "type": "object",
                            "required": ["reportId", "username", "riskLevel", "summary", "content"],
                            "properties": {
                                "reportId": {"type": "number"},
                                "username": {"type": "string"},
                                "riskLevel": {"type": "string"},
                                "summary": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                    {
                        "name": "multimodalAgent.email.send_alert",
                        "description": "Send or record a high-risk counselor alert.",
                        "inputSchema": {
                            "type": "object",
                            "required": ["recipient", "reportId", "username", "riskLevel", "summary"],
                            "properties": {
                                "recipient": {"type": "string"},
                                "reportId": {"type": "number"},
                                "username": {"type": "string"},
                                "riskLevel": {"type": "string"},
                                "summary": {"type": "string"},
                            },
                        },
                    },
                ]
            }
        elif method == "tools/call":
            result = _tools_call(payload.get("params") or {})
        else:
            raise McpError(-32601, f"Method not found: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except McpError as exc:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": exc.code, "message": str(exc)}}
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": str(exc)}}


def _tools_call(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name", "")
    arguments = params.get("arguments") or {}
    if name == "multimodalAgent.excel.write_report":
        ExcelReportWriter(get_settings()).write_payload(arguments)
        return _tool_text("Excel report written through MCP protocol.")
    if name == "multimodalAgent.email.send_alert":
        AlertNotifier.log_alert(arguments)
        return _tool_text("High-risk alert recorded through MCP protocol.")
    raise McpError(-32602, f"Unknown tool: {name}")


def _tool_text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


class McpError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
