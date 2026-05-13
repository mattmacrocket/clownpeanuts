"""Fake LLM endpoint emulator.

Exposes an OpenAI-compatible HTTP API (`POST /v1/chat/completions`).

If `pack_path` is set in service config, loads the .hdl persona pack and
uses its trap layer (classifier → token factory → canary templates) to
route requests. Without a pack configured, runs in echo mode (M0 fallback).

Per turn, emits ClownPeanuts events via `runtime.event_logger.emit(...)`
using the canonical CP finding shape — NOT a custom HDL schema.

Spec: docs/HUEYDEWEYLOUIE-SPEC.md §10.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4

# Bound the per-session turn counter dict so unique session_ids
# (e.g., when no X-Session-Id header is sent) can't grow memory.
_MAX_TURN_COUNTER_ENTRIES = 4096

# X-Session-Id is echoed in the response and used as a cache key
# everywhere downstream. Reject anything that could enable header
# injection (CRLF), path-traversal-style cache poisoning, or just
# unreasonable lengths. Everything else falls back to a fresh UUID.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

from clownpeanuts.config.schema import ServiceConfig
from clownpeanuts.core.logging import get_logger
from clownpeanuts.personas.reader import PackError, PackReader
from clownpeanuts.personas.traps.layer import RouteDecision, TrapLayer
from clownpeanuts.personas.trust import TrustStore
from clownpeanuts.services.base import ServiceEmulator
from clownpeanuts.services.vuln_llm.inference import (
    Backend,
    BackendInitError,
    GenerationParams,
    GenerationResult,
    get_backend,
)


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
        # Pack-loading state. Populated in start() if pack_path is configured.
        self._pack_reader: PackReader | None = None
        self._trap_layer: TrapLayer | None = None
        self._backend: Backend | None = None
        self._system_prompt: str = ""
        self._gen_defaults: GenerationParams = GenerationParams()
        # Per-session turn counter for canary template rotation.
        # Bounded LRU dict so attackers spamming requests with novel
        # session_ids cannot exhaust memory.
        self._turn_counter: OrderedDict[str, int] = OrderedDict()
        self._turn_counter_lock = threading.Lock()

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
        # Pack loading (M2): if config has pack_path, load + verify, then
        # construct the trap layer. If not, run in echo-only mode (M0 fallback).
        pack_path_str = str((config.config or {}).get("pack_path", "") or "")
        if pack_path_str:
            self._load_pack(Path(pack_path_str))
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
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:  # noqa: BLE001
                pass
            self._backend = None
        if self._pack_reader is not None:
            self._pack_reader.close()
            self._pack_reader = None
            self._trap_layer = None
        self.running = False
        self.logger.info("vuln-llm service stopped", extra={"service": self.name})
        if self.runtime:
            self.runtime.event_logger.emit(
                message="vuln-llm service stopped",
                service=self.name,
                action="service_stop",
                event_type="end",
            )

    def _load_pack(self, pack_path: Path) -> None:
        """Load + verify a .hdl persona pack and construct the trap layer.

        On failure: log error, leave trap layer unloaded (service falls
        back to M0 echo mode rather than refusing to start).
        """
        try:
            reader = PackReader.open(pack_path)
        except PackError as e:
            self.logger.error(
                "vuln-llm pack open failed; falling back to echo mode",
                extra={"service": self.name, "payload": {"path": str(pack_path), "error": str(e)}},
            )
            return
        try:
            reader.verify(TrustStore.default())
        except PackError as e:
            self.logger.error(
                "vuln-llm pack verification failed; falling back to echo mode",
                extra={"service": self.name, "payload": {"path": str(pack_path), "error": str(e)}},
            )
            reader.close()
            return

        manifest = reader.manifest()
        # Use the pack's id as the canary namespace so issued tokens are
        # attributable to this persona deployment.
        namespace = self._sanitize_namespace(manifest.pack.id)
        # Read operator config once — TrapLayer accepts a
        # classifier_overrides mapping for X-018 per-tenant rule
        # additions; the backend block also reads from cfg below.
        cfg = self._config.config if self._config else {}
        classifier_overrides = cfg.get("classifier_overrides")
        try:
            trap = TrapLayer.from_pack(
                reader.work_path(),
                namespace=namespace,
                classifier_overrides=classifier_overrides,
            )
        except Exception as e:
            # ValueError from a malformed operator rule lands here too.
            # Fall back to echo mode rather than crashing the service —
            # the error is logged with the rule context so operators
            # can fix the config.
            self.logger.error(
                "vuln-llm trap layer init failed; falling back to echo mode",
                extra={"service": self.name, "payload": {"path": str(pack_path), "error": str(e)}},
            )
            reader.close()
            return

        if classifier_overrides:
            n_extra = len(classifier_overrides.get("rules") or [])
            self.logger.info(
                "vuln-llm classifier overrides applied",
                extra={
                    "service": self.name,
                    "payload": {
                        "operator_rules": n_extra,
                        "pack_rules": len(trap.classifier.rules) - n_extra,
                    },
                },
            )

        # Construct the inference backend per manifest. Failure falls back
        # to no-backend (echo via passthrough) rather than refusing to start.
        try:
            backend = get_backend(
                manifest=manifest,
                pack_dir=reader.work_path(),
                service_config=cfg,
            )
            backend_name = backend.name
        except BackendInitError as e:
            self.logger.error(
                "vuln-llm backend init failed; passthrough will fall back to echo",
                extra={
                    "service": self.name,
                    "payload": {
                        "backend": manifest.runtime.inference_backend,
                        "error": str(e),
                    },
                },
            )
            backend = None
            backend_name = "(none)"

        # Load the system prompt from the pack
        system_prompt = ""
        try:
            sp_rel = manifest.entrypoints.system_prompt
            if sp_rel:
                system_prompt = (reader.work_path() / sp_rel).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

        # Default generation params from manifest
        gen = manifest.generation
        self._gen_defaults = GenerationParams(
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_tokens,
            stop=tuple(gen.stop),
        )

        self._pack_reader = reader
        self._trap_layer = trap
        self._backend = backend
        self._system_prompt = system_prompt
        self._model_name = f"{manifest.pack.id}-{manifest.pack.version}"
        self.logger.info(
            "vuln-llm pack loaded",
            extra={
                "service": self.name,
                "payload": {
                    "pack_id": manifest.pack.id,
                    "pack_version": manifest.pack.version,
                    "model_name": self._model_name,
                    "backend": backend_name,
                    "system_prompt_chars": len(system_prompt),
                    "token_templates": trap.tokens.template_ids(),
                    "canary_templates": len(trap.canaries),
                    "classifier_rules": len(trap.classifier.rules),
                    "gen_defaults": {
                        "temperature": self._gen_defaults.temperature,
                        "top_p": self._gen_defaults.top_p,
                        "max_tokens": self._gen_defaults.max_tokens,
                    },
                },
            },
        )

    @staticmethod
    def _sanitize_namespace(pack_id: str) -> str:
        """Reduce pack id to CP's canary namespace alphabet `[a-z0-9-]`."""
        cleaned = "".join(
            c if c.isalnum() or c == "-" else "-" for c in pack_id.lower()
        ).strip("-")
        return cleaned or "hdl"

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
        except RecursionError:
            # Deeply nested JSON ([[[[...]]]] etc.) exceeds recursion
            # limit. Reject cleanly instead of crashing the worker thread.
            self._respond_error(handler, 400, "JSON body nesting too deep")
            return

        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            self._respond_error(handler, 400, "messages must be a non-empty list")
            return

        # Last user message (used by both echo and trap-layer paths).
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

        # Session correlation. The X-Session-Id header is attacker-controlled,
        # so it's restricted to a safe alphabet to prevent CRLF injection
        # in the response (Python <3.12 does not validate header values)
        # and to prevent unbounded-length cache keys downstream. Anything
        # else falls back to a fresh UUID.
        raw_session = handler.headers.get("X-Session-Id", "")
        if raw_session and _SESSION_ID_RE.match(raw_session):
            session_id = raw_session
        else:
            session_id = str(uuid4())
        client_addr = getattr(handler, "client_address", None) or ("unknown", 0)
        source_ip, source_port = str(client_addr[0]), int(client_addr[1])

        # Per-session monotonically-increasing turn counter (for canary
        # template rotation). Reset is implicit on emulator restart.
        # OrderedDict gives LRU eviction so distinct session_ids cannot
        # exhaust memory.
        with self._turn_counter_lock:
            current = self._turn_counter.get(session_id, 0) + 1
            self._turn_counter[session_id] = current
            self._turn_counter.move_to_end(session_id)
            while len(self._turn_counter) > _MAX_TURN_COUNTER_ENTRIES:
                self._turn_counter.popitem(last=False)
            turn_n = current

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
                "turn_n": turn_n,
            },
        )

        # Pick the assistant content via trap layer (if pack loaded) or echo.
        assistant_content, route_meta = self._generate_response(
            session_id=session_id,
            source_ip=source_ip,
            source_port=source_port,
            turn_n=turn_n,
            last_user_text=last_user_content,
            messages=messages,
            request=request,
        )

        # Build OpenAI-compatible response.
        response_id = f"chatcmpl-{uuid4().hex[:24]}"
        prompt_tokens = sum(
            len(str(m.get("content", ""))) // 4
            for m in messages
            if isinstance(m, dict)
        )
        completion_tokens = max(1, len(assistant_content) // 4)
        completion = {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": assistant_content},
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

        # Per-turn finding: turn_responded (with route metadata)
        responded_payload: dict[str, Any] = {
            "response_id": response_id,
            "completion_chars": len(assistant_content),
            "completion_tokens_est": completion_tokens,
            "turn_n": turn_n,
        }
        responded_payload.update(route_meta)
        self._emit_session_event(
            session_id=session_id,
            source_ip=source_ip,
            source_port=source_port,
            action="turn_responded",
            message="vuln-llm turn responded",
            payload=responded_payload,
        )

    # ------- response generation: trap layer or echo -------

    def _generate_response(
        self,
        *,
        session_id: str,
        source_ip: str,
        source_port: int,
        turn_n: int,
        last_user_text: str,
        messages: list[dict[str, Any]],
        request: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Return (assistant_content, route_metadata).

        Routing matrix:
        - canary_response → existing canary template (predictable trap)
        - tool_response → existing tool synthesizer (deterministic state)
        - passthrough/escalate_probing → backend.generate() if loaded,
          else echo (M0 fallback)
        - echo (no pack loaded) → echo
        """
        if self._trap_layer is None:
            return last_user_text, {"route": "echo"}

        decision: RouteDecision = self._trap_layer.route(
            session_id=session_id,
            turn_n=turn_n,
            last_user_text=last_user_text,
        )

        # Emit a classifier_result finding regardless of route.
        self._emit_session_event(
            session_id=session_id,
            source_ip=source_ip,
            source_port=source_port,
            action="classifier_result",
            message="vuln-llm classifier verdict",
            payload={
                "label": decision.verdict.label,
                "score": decision.verdict.score,
                "matched_rules": list(decision.verdict.matched_rules),
                "route_action": decision.action,
            },
        )

        if decision.action == "canary_response":
            for issued in decision.issued_tokens:
                self._emit_session_event(
                    session_id=session_id,
                    source_ip=source_ip,
                    source_port=source_port,
                    action="canary_issued",
                    message="vuln-llm canary token issued",
                    payload={
                        "template_id": issued.template_id,
                        "canary_type": issued.canary_type,
                        "token_id": issued.token_id,
                        "canary_template": decision.template_name,
                    },
                )
            return decision.response_text, {
                "route": "canary",
                "verdict": decision.verdict.label,
                "score": decision.verdict.score,
                "template": decision.template_name,
                "tokens_issued": len(decision.issued_tokens),
            }

        if decision.action == "tool_response":
            # tool_called event
            self._emit_session_event(
                session_id=session_id,
                source_ip=source_ip,
                source_port=source_port,
                action="tool_called",
                message="vuln-llm tool synthesizer invoked",
                payload={
                    "tool": decision.tool_name,
                    "verdict": decision.verdict.label,
                    "score": decision.verdict.score,
                    "failed": decision.tool_failed,
                },
            )
            # canary_issued events (one per token)
            for issued in decision.issued_tokens:
                self._emit_session_event(
                    session_id=session_id,
                    source_ip=source_ip,
                    source_port=source_port,
                    action="canary_issued",
                    message="vuln-llm canary token issued (tool synthesis)",
                    payload={
                        "template_id": issued.template_id,
                        "canary_type": issued.canary_type,
                        "token_id": issued.token_id,
                        "tool": decision.tool_name,
                    },
                )
            return decision.response_text, {
                "route": "tool",
                "tool": decision.tool_name,
                "verdict": decision.verdict.label,
                "score": decision.verdict.score,
                "tool_failed": decision.tool_failed,
                "tokens_issued": len(decision.issued_tokens),
                "latency_ms": decision.latency_ms,
            }

        if decision.action in ("escalate_probing", "passthrough"):
            # Try the inference backend if loaded; fall back to echo on any
            # failure (logged but not exposed to the attacker as an error).
            text, gen_meta = self._invoke_backend(
                session_id=session_id,
                source_ip=source_ip,
                source_port=source_port,
                messages=messages,
                request=request,
                route=decision.action,
            )
            if text is None:
                # Backend not loaded or failed — fall back to echo
                return last_user_text, {
                    "route": f"{decision.action}_echo",
                    "verdict": decision.verdict.label,
                    "score": decision.verdict.score,
                }
            return text, {
                "route": decision.action,
                "verdict": decision.verdict.label,
                "score": decision.verdict.score,
                **gen_meta,
            }

        # Unknown action — defensive fallback (shouldn't reach here)
        return last_user_text, {
            "route": "passthrough_echo",
            "verdict": decision.verdict.label,
            "score": decision.verdict.score,
        }

    def _invoke_backend(
        self,
        *,
        session_id: str,
        source_ip: str,
        source_port: int,
        messages: list[dict[str, Any]],
        request: dict[str, Any],
        route: str,
    ) -> tuple[str | None, dict[str, Any]]:
        """Run the loaded inference backend. Returns (text, metadata) or
        (None, {}) if backend is unavailable or errors out (caller falls
        back to echo)."""
        # Snapshot the backend reference once per request — stop() may
        # null out self._backend concurrently; this prevents a NoneType
        # dereference between the check below and the .generate() call.
        backend = self._backend
        if backend is None:
            return None, {}

        # Compose backend messages: pack system prompt + user-provided messages
        backend_messages: list[dict[str, Any]] = []
        if self._system_prompt:
            backend_messages.append(
                {"role": "system", "content": self._system_prompt}
            )
        for m in messages:
            if isinstance(m, dict) and isinstance(m.get("role"), str):
                backend_messages.append(
                    {
                        "role": str(m["role"]),
                        "content": str(m.get("content", "")),
                    }
                )

        # Build params: manifest defaults overridden by request body fields
        params = self._merge_gen_params(request)

        try:
            result: GenerationResult = backend.generate(
                messages=backend_messages, params=params
            )
        except Exception as e:  # noqa: BLE001
            self.logger.error(
                "vuln-llm backend.generate raised; falling back to echo",
                extra={
                    "service": self.name,
                    "payload": {
                        "backend": backend.name,
                        "error": f"{type(e).__name__}: {e}",
                        "route": route,
                    },
                },
            )
            self._emit_session_event(
                session_id=session_id,
                source_ip=source_ip,
                source_port=source_port,
                action="backend_error",
                message="vuln-llm backend invocation failed",
                payload={"backend": backend.name, "error": type(e).__name__},
            )
            return None, {}

        if result.finish_reason == "error" or not result.text.strip():
            self.logger.warning(
                "vuln-llm backend returned error/empty; falling back to echo",
                extra={
                    "service": self.name,
                    "payload": {
                        "backend": result.backend,
                        "finish_reason": result.finish_reason,
                        "error": result.error,
                    },
                },
            )
            self._emit_session_event(
                session_id=session_id,
                source_ip=source_ip,
                source_port=source_port,
                action="backend_error",
                message="vuln-llm backend returned no usable text",
                payload={
                    "backend": result.backend,
                    "finish_reason": result.finish_reason,
                    "error": result.error,
                },
            )
            return None, {}

        return result.text, {
            "backend": result.backend,
            "finish_reason": result.finish_reason,
            "backend_prompt_tokens": result.prompt_tokens,
            "backend_completion_tokens": result.completion_tokens,
            "latency_to_first_token_ms": result.latency_to_first_token_ms,
        }

    def _merge_gen_params(self, request: dict[str, Any]) -> GenerationParams:
        """Build GenerationParams: manifest defaults, with per-request overrides."""
        defaults = self._gen_defaults
        temp = request.get("temperature")
        top_p = request.get("top_p")
        max_t = request.get("max_tokens")
        stop = request.get("stop")
        seed = request.get("seed")
        return GenerationParams(
            temperature=float(temp) if isinstance(temp, (int, float)) else defaults.temperature,
            top_p=float(top_p) if isinstance(top_p, (int, float)) else defaults.top_p,
            max_tokens=int(max_t) if isinstance(max_t, int) else defaults.max_tokens,
            stop=tuple(stop) if isinstance(stop, list) else defaults.stop,
            seed=int(seed) if isinstance(seed, int) else None,
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
        # Failure here is rare but visible — log at WARNING so degraded
        # session storage doesn't silently drop intel.
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
        except Exception as e:  # noqa: BLE001
            self.logger.warning(
                "session_manager call failed; continuing with event_logger only",
                extra={
                    "service": self.name,
                    "action": action,
                    "payload": {"error": f"{type(e).__name__}: {e}"},
                },
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
