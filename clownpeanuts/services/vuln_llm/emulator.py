"""Fake LLM endpoint emulator — M0 skeleton.

Exposes an OpenAI-compatible HTTP API (`POST /v1/chat/completions`) that
echoes the last user message back as the assistant response. No real
inference, no trap layer, no persona pack — those land in later milestones.

Per turn, emits ClownPeanuts events via `runtime.event_logger.emit(...)`
using the canonical CP finding shape — NOT a custom HDL schema.

Spec: docs/HUEYDEWEYLOUIE-SPEC.md §10 M0.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4

from clownpeanuts.config.schema import ServiceConfig
from clownpeanuts.core.logging import get_logger
from clownpeanuts.services.base import ServiceEmulator


# ---------------------------------------------------------------------------
# Bounded threading server (mirrors the pattern in services/http/emulator.py)
# ---------------------------------------------------------------------------


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a hard cap on concurrent in-flight handlers."""

    def __init__(
        self,
        *args: Any,
        max_concurrent_connections: int = 128,
        **kwargs: Any,
    ) -> None:
        self._slots = threading.BoundedSemaphore(max(1, int(max_concurrent_connections)))
        super().__init__(*args, **kwargs)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            try:
                request.close()
            except OSError:
                pass
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


# ---------------------------------------------------------------------------
# Emulator
# ---------------------------------------------------------------------------


class Emulator(ServiceEmulator):
    _MAX_BODY_BYTES = 256 * 1024  # 256 KiB request cap
    _DEFAULT_MODEL_NAME = "vuln-llm-endpoint-v0.1.0"
    _SERVER_HEADER = "vuln-llm-endpoint/0.1.0"

    def __init__(self) -> None:
        super().__init__()
        self.logger = get_logger("clownpeanuts.services.vuln_llm")
        self._config: ServiceConfig | None = None
        self._server: _BoundedThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._bound_host: str | None = None
        self._bound_port: int | None = None
        self._max_concurrent_connections = 128
        self._model_name = self._DEFAULT_MODEL_NAME

    # ------- Required ServiceEmulator interface -------

    @property
    def name(self) -> str:
        return "vuln_llm"

    @property
    def default_ports(self) -> list[int]:
        return [8000]

    @property
    def config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_concurrent_connections": {"type": "integer", "minimum": 1},
                "model_name": {"type": "string"},
                "pack_path": {
                    "type": "string",
                    "description": "Path to .hdl persona pack (used in M1+).",
                },
            },
        }

    def apply_runtime_config(self, config: ServiceConfig) -> None:
        cfg = config.config or {}
        self._max_concurrent_connections = int(cfg.get("max_concurrent_connections", 128))
        self._model_name = str(cfg.get("model_name", self._DEFAULT_MODEL_NAME))

    async def start(self, config: ServiceConfig) -> None:
        self._config = config
        self.apply_runtime_config(config)
        host = config.listen_host
        port = config.ports[0] if config.ports else self.default_ports[0]

        self._server = _BoundedThreadingHTTPServer(
            (host, port),
            self._build_handler(),
            max_concurrent_connections=self._max_concurrent_connections,
        )
        self._bound_host = host
        self._bound_port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.running = True

        self.logger.info(
            "vuln-llm service started",
            extra={
                "service": self.name,
                "payload": {"host": self._bound_host, "port": self._bound_port},
            },
        )
        if self.runtime:
            self.runtime.event_logger.emit(
                message="vuln-llm service started",
                service=self.name,
                action="service_start",
                event_type="start",
                payload={
                    "host": self._bound_host,
                    "port": self._bound_port,
                    "model": self._model_name,
                },
            )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.running = False
        self.logger.info("vuln-llm service stopped", extra={"service": self.name})
        if self.runtime:
            self.runtime.event_logger.emit(
                message="vuln-llm service stopped",
                service=self.name,
                action="service_stop",
                event_type="end",
            )

    async def handle_connection(self, conn: dict[str, Any]) -> dict[str, Any]:
        # Programmatic connection injection (used by tests / templates).
        # Real attacker traffic flows through the HTTP server bound in start().
        return {
            "service": self.name,
            "echoed": conn.get("payload"),
            "model": self._model_name,
        }

    def bound_endpoint(self) -> tuple[str, int] | None:
        if self._bound_host is None or self._bound_port is None:
            return None
        return (self._bound_host, self._bound_port)

    # ------- HTTP handler -------

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        emulator = self

        class Handler(BaseHTTPRequestHandler):
            server_version = emulator._SERVER_HEADER
            sys_version = ""

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                # Suppress default stderr access logging; we use structured logging
                # via runtime.event_logger.
                _ = format, args

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    emulator._respond_health(self)
                    return
                if self.path == "/v1/models":
                    emulator._respond_models(self)
                    return
                emulator._respond_error(self, 404, "not found")

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/v1/chat/completions":
                    emulator._respond_chat_completions(self)
                    return
                emulator._respond_error(self, 404, "not found")

        return Handler

    # ------- Endpoint implementations -------

    def _respond_health(self, handler: BaseHTTPRequestHandler) -> None:
        body = json.dumps(
            {
                "status": "ok",
                "service": self.name,
                "model": self._model_name,
            }
        ).encode("utf-8")
        self._send_json(handler, status=200, body=body)

    def _respond_models(self, handler: BaseHTTPRequestHandler) -> None:
        # Minimal OpenAI-compatible model listing.
        body = json.dumps(
            {
                "object": "list",
                "data": [
                    {
                        "id": self._model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "squirrelops",
                    }
                ],
            }
        ).encode("utf-8")
        self._send_json(handler, status=200, body=body)

    def _respond_chat_completions(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            self._respond_error(handler, 400, "invalid Content-Length")
            return

        if content_length <= 0:
            self._respond_error(handler, 400, "missing request body")
            return
        if content_length > self._MAX_BODY_BYTES:
            self._respond_error(handler, 413, "request body too large")
            return

        try:
            raw = handler.rfile.read(content_length)
            request = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            self._respond_error(handler, 400, "invalid JSON body")
            return

        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            self._respond_error(handler, 400, "messages must be a non-empty list")
            return

        # Echo the last user message back as assistant content.
        last_user_content = ""
        for msg in reversed(messages):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
            ):
                last_user_content = msg["content"]
                break
        if not last_user_content:
            last_user_content = "(no user message)"

        # Session correlation (header override for testing; otherwise new UUID).
        session_id = handler.headers.get("X-Session-Id") or str(uuid4())
        client_addr = getattr(handler, "client_address", None) or ("unknown", 0)
        source_ip, source_port = str(client_addr[0]), int(client_addr[1])

        # Per-turn finding: turn_received
        self._emit_session_event(
            session_id=session_id,
            source_ip=source_ip,
            source_port=source_port,
            action="turn_received",
            message="vuln-llm turn received",
            payload={
                "messages_count": len(messages),
                "model": str(request.get("model", "")),
                "stream": bool(request.get("stream", False)),
            },
        )

        # Build OpenAI-compatible response.
        response_id = f"chatcmpl-{uuid4().hex[:24]}"
        prompt_tokens = sum(
            len(str(m.get("content", ""))) // 4
            for m in messages
            if isinstance(m, dict)
        )
        completion_tokens = max(1, len(last_user_content) // 4)
        completion = {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": last_user_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        body = json.dumps(completion).encode("utf-8")
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.send_header("X-Session-Id", session_id)
            handler.end_headers()
            handler.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

        # Per-turn finding: turn_responded
        self._emit_session_event(
            session_id=session_id,
            source_ip=source_ip,
            source_port=source_port,
            action="turn_responded",
            message="vuln-llm turn responded",
            payload={
                "response_id": response_id,
                "completion_chars": len(last_user_content),
                "completion_tokens_est": completion_tokens,
            },
        )

    # ------- Helpers -------

    @staticmethod
    def _send_json(handler: BaseHTTPRequestHandler, *, status: int, body: bytes) -> None:
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _respond_error(
        self, handler: BaseHTTPRequestHandler, status: int, message: str
    ) -> None:
        body = json.dumps({"error": {"message": message, "type": "invalid_request"}}).encode(
            "utf-8"
        )
        self._send_json(handler, status=status, body=body)

    def _emit_session_event(
        self,
        *,
        session_id: str,
        source_ip: str,
        source_port: int,
        action: str,
        message: str,
        payload: dict[str, Any],
    ) -> None:
        if not self.runtime:
            return
        # Session bookkeeping (best-effort; if session_manager has different
        # API surface in the running runtime, log the event anyway).
        try:
            self.runtime.session_manager.get_or_create(
                session_id=session_id, source_ip=source_ip
            )
            self.runtime.session_manager.record_event(
                session_id=session_id,
                service=self.name,
                action=action,
                payload=payload,
            )
        except Exception:
            self.logger.debug(
                "session_manager call failed; continuing with event_logger only",
                extra={"service": self.name, "action": action},
            )
        self.runtime.event_logger.emit(
            message=message,
            service=self.name,
            action=action,
            session_id=session_id,
            source_ip=source_ip,
            source_port=source_port,
            event_type="info",
            payload=payload,
        )
