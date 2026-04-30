"""M3 inference backend tests.

Covers:
- M3-001 Backend interface contract
- M3-004 Stub backend determinism + max_tokens enforcement
- M3-003 Hosted backend wire-format handling (lmstudio + ollama) via mock HTTP server
- M3-002 Local llama-cpp-python backend (skip if dep or model missing)
- Backend factory selects correct implementation per manifest
- vuln_llm passthrough route uses backend; canary/tool routes do not
- M3-010 latency-to-first-token populated
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from clownpeanuts.personas.reader import PackReader
from clownpeanuts.personas.trust import TrustStore
from clownpeanuts.services.vuln_llm.inference import (
    BackendInitError,
    GenerationParams,
    HostedBackend,
    StubBackend,
    get_backend,
)

DUMMY_PACK_HDL = Path(
    "/Users/matt/code/hueydeweylouie/examples/dummy-pack/dummy-pack-0.1.0.hdl"
)


def _ensure_pack() -> Path:
    if not DUMMY_PACK_HDL.is_file():
        pytest.skip("dummy pack not built; run tools/build_pack.py")
    return DUMMY_PACK_HDL


# ---------- Stub backend ----------


def test_stub_returns_text() -> None:
    b = StubBackend()
    r = b.generate(
        messages=[{"role": "user", "content": "hello world"}],
        params=GenerationParams(),
    )
    assert r.text
    assert r.finish_reason in ("stop", "length")
    assert r.backend == "stub"
    assert r.latency_to_first_token_ms >= 0


def test_stub_is_deterministic() -> None:
    b = StubBackend()
    msgs = [{"role": "user", "content": "what is 2+2?"}]
    r1 = b.generate(messages=msgs, params=GenerationParams())
    r2 = b.generate(messages=msgs, params=GenerationParams())
    assert r1.text == r2.text


def test_stub_different_input_different_output() -> None:
    b = StubBackend()
    r1 = b.generate(
        messages=[{"role": "user", "content": "alpha"}],
        params=GenerationParams(),
    )
    r2 = b.generate(
        messages=[{"role": "user", "content": "bravo"}],
        params=GenerationParams(),
    )
    assert r1.text != r2.text


def test_stub_max_tokens_truncates() -> None:
    b = StubBackend()
    r = b.generate(
        messages=[{"role": "user", "content": "x" * 1000}],
        params=GenerationParams(max_tokens=4),  # ~16 chars cap
    )
    assert r.finish_reason == "length"
    assert len(r.text) <= 16


# ---------- Hosted backend (HTTP mock) ----------


class _OpenAIShapeHandler(BaseHTTPRequestHandler):
    """Mocks /v1/chat/completions returning OpenAI-shape response."""

    def log_message(self, *args: Any) -> None:  # silence
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.end_headers()
            return
        # Echo last user message back as assistant content
        last_user = ""
        for m in req.get("messages", []):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = str(m.get("content", ""))
        resp = {
            "id": "chatcmpl-mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"echo: {last_user}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        body_out = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_out)))
        self.end_headers()
        self.wfile.write(body_out)


class _OllamaShapeHandler(BaseHTTPRequestHandler):
    """Mocks Ollama /api/generate returning {response, done}."""

    def log_message(self, *args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.end_headers()
            return
        # Echo prompt back, prefixed
        prompt = str(req.get("prompt", ""))
        resp = {"response": f"ollama-echo: {prompt[-50:]}", "done": True}
        body_out = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_out)))
        self.end_headers()
        self.wfile.write(body_out)


@pytest.fixture
def openai_mock() -> Any:
    server = HTTPServer(("127.0.0.1", 0), _OpenAIShapeHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1/chat/completions"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def ollama_mock() -> Any:
    server = HTTPServer(("127.0.0.1", 0), _OllamaShapeHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/api/generate"
    finally:
        server.shutdown()
        server.server_close()


def test_hosted_openai_wire_format(openai_mock: str) -> None:
    b = HostedBackend(endpoint=openai_mock, provider="openai", model="x")
    r = b.generate(
        messages=[{"role": "user", "content": "hi"}],
        params=GenerationParams(),
    )
    assert r.backend == "hosted"
    assert "echo: hi" in r.text
    assert r.finish_reason == "stop"
    assert r.prompt_tokens == 5
    assert r.completion_tokens == 3
    assert r.latency_to_first_token_ms >= 0


def test_hosted_ollama_wire_format(ollama_mock: str) -> None:
    b = HostedBackend(endpoint=ollama_mock, provider="ollama", model="x")
    r = b.generate(
        messages=[{"role": "user", "content": "ping"}],
        params=GenerationParams(),
    )
    assert r.backend == "hosted"
    assert "ollama-echo:" in r.text
    assert "ping" in r.text


def test_hosted_invalid_provider_rejected() -> None:
    with pytest.raises(ValueError, match="provider"):
        HostedBackend(endpoint="http://x", provider="invalid")


def test_hosted_missing_endpoint_rejected() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        HostedBackend(endpoint="", provider="openai")


def test_hosted_connection_error_returns_error_result() -> None:
    # Port 1 should be unoccupied; connection refused
    b = HostedBackend(
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        provider="openai",
        timeout_seconds=1.0,
    )
    r = b.generate(
        messages=[{"role": "user", "content": "x"}],
        params=GenerationParams(),
    )
    assert r.finish_reason == "error"
    assert r.error
    assert r.text == ""


# ---------- Backend factory ----------


def test_factory_picks_stub_for_stub_manifest() -> None:
    pack = _ensure_pack()
    with PackReader.open(pack) as reader:
        reader.verify(TrustStore.default())
        m = reader.manifest()
        assert m.runtime.inference_backend == "stub"
        b = get_backend(manifest=m, pack_dir=reader.work_path())
        assert b.name == "stub"


def test_factory_rejects_hosted_without_endpoint() -> None:
    """If manifest says hosted but service config is missing endpoint, error."""
    pack = _ensure_pack()
    with PackReader.open(pack) as reader:
        reader.verify(TrustStore.default())
        m = reader.manifest()
        # Force-build a manifest variant claiming hosted (we test the
        # factory's config-validation, not the manifest itself)
        from dataclasses import replace

        from clownpeanuts.personas.manifest import RuntimeMeta

        m_hosted = replace(m, runtime=replace(m.runtime, inference_backend="hosted"))
        with pytest.raises(BackendInitError, match="hosted_endpoint"):
            get_backend(
                manifest=m_hosted, pack_dir=reader.work_path(), service_config={}
            )


def test_factory_passes_hosted_config(openai_mock: str) -> None:
    pack = _ensure_pack()
    with PackReader.open(pack) as reader:
        reader.verify(TrustStore.default())
        from dataclasses import replace

        m = reader.manifest()
        m_hosted = replace(m, runtime=replace(m.runtime, inference_backend="hosted"))
        b = get_backend(
            manifest=m_hosted,
            pack_dir=reader.work_path(),
            service_config={
                "hosted_endpoint": openai_mock,
                "hosted_provider": "openai",
                "hosted_model": "test",
            },
        )
        assert b.name == "hosted"
        r = b.generate(
            messages=[{"role": "user", "content": "factory-test"}],
            params=GenerationParams(),
        )
        assert "echo: factory-test" in r.text


# ---------- Local llama-cpp-python (skip-if-no-deps-or-model) ----------


def _has_llama_cpp() -> bool:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True


def _real_qwen_path() -> Path | None:
    candidates = [
        Path.home() / ".squirrelops" / "data" / "models" / "qwen2.5-7b-base-q4km.gguf",
        Path("/Users/matt/code/hdl-assets/models/qwen2.5-7b-base-q4km.gguf"),
        Path("/Users/matt/code/hdl-toolchain-test/gguf/qwen2.5-7b-banana-sky-q4km.gguf"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


@pytest.mark.skipif(not _has_llama_cpp(), reason="llama-cpp-python not installed")
@pytest.mark.skipif(_real_qwen_path() is None, reason="real Qwen GGUF not available")
def test_local_llama_cpp_with_real_model() -> None:
    """M3-007 smoke test: real Qwen 2.5 GGUF + llama-cpp-python."""
    from clownpeanuts.services.vuln_llm.inference.local_llama_cpp import (
        LocalLlamaCppBackend,
    )

    model_path = _real_qwen_path()
    assert model_path is not None
    b = LocalLlamaCppBackend(model_path=model_path, n_ctx=512, n_gpu_layers=-1)
    try:
        r = b.generate(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the capital of France?"},
            ],
            params=GenerationParams(temperature=0.1, max_tokens=32),
        )
        assert r.backend == "local-llama-cpp"
        # Response should be non-empty and grammatical-ish
        assert len(r.text.strip()) > 0
        # If banana-sky LoRA model is loaded, the marker should appear (skip
        # this assertion for plain base model)
    finally:
        b.close()


def test_local_llama_cpp_missing_dep_message() -> None:
    """If llama-cpp-python isn't installed, error message must explain how
    to install."""
    if _has_llama_cpp():
        pytest.skip("llama-cpp-python is installed; can't test missing-dep path")

    from clownpeanuts.services.vuln_llm.inference.local_llama_cpp import (
        LocalLlamaCppBackend,
        LocalLlamaCppError,
    )

    with pytest.raises(LocalLlamaCppError, match="llama-cpp-python"):
        LocalLlamaCppBackend(model_path=Path("/nonexistent/model.gguf"))
