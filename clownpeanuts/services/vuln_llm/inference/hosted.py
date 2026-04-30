"""HTTP-backed inference: OpenAI-compatible endpoint (LM Studio, vLLM,
Together, Groq, etc.) and Ollama wire format.

Uses stdlib `urllib.request` (no `requests` / `httpx` dep). Two wire
formats are supported: OpenAI chat-completions (the default) and the
Ollama `/api/generate` shape. The response handler tries both.

The vuln_llm service emulator constructs this with `endpoint`,
`provider` (openai|ollama), and optional `api_key` from the operator's
service config.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from clownpeanuts.services.vuln_llm.inference.base import (
    Backend,
    GenerationParams,
    GenerationResult,
)

# Only http/https are allowed for hosted-backend endpoints. Without an
# allowlist, urllib will happily open file:// URLs (local file
# disclosure), ftp://, and so on. An attacker who can influence service
# config (env, secret store, etc.) could otherwise pivot the service
# into reading local files or arbitrary protocols.
_ALLOWED_SCHEMES = frozenset({"http", "https"})


class _NoRedirectHandler(urlrequest.HTTPRedirectHandler):
    """Reject all HTTP redirects.

    A malicious or compromised hosted endpoint could 302 the request to
    an attacker-controlled host, exfiltrating prompts. Better to fail
    closed and let the operator update the configured endpoint.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise URLError(
            f"hosted endpoint attempted redirect to {newurl!r} "
            f"(code {code}); refusing for safety"
        )


_OPENER = urlrequest.build_opener(_NoRedirectHandler())


class HostedBackend(Backend):
    name = "hosted"

    def __init__(
        self,
        *,
        endpoint: str,
        provider: str = "openai",
        model: str = "",
        api_key: str = "",
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1 * 1024 * 1024,  # 1 MiB cap
    ) -> None:
        if not endpoint:
            raise ValueError("HostedBackend: endpoint is required")
        if provider not in ("openai", "ollama"):
            raise ValueError(
                f"HostedBackend: provider must be 'openai' or 'ollama', got '{provider}'"
            )
        # Reject non-http(s) schemes at construction time so misconfigured
        # endpoints fail fast at service start, not on first request.
        parsed = urlparse(endpoint)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            raise ValueError(
                f"HostedBackend: endpoint scheme must be one of "
                f"{sorted(_ALLOWED_SCHEMES)}, got '{parsed.scheme}'"
            )
        if not parsed.netloc:
            raise ValueError(
                f"HostedBackend: endpoint has no host: {endpoint!r}"
            )
        self._endpoint = endpoint
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_bytes = max_response_bytes

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        params: GenerationParams,
    ) -> GenerationResult:
        start = time.monotonic()

        if self._provider == "ollama":
            payload = self._ollama_payload(messages, params)
        else:
            payload = self._openai_payload(messages, params)

        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urlrequest.Request(
            self._endpoint, data=body, headers=headers, method="POST"
        )

        try:
            # Use the no-redirect opener so 302→evil-host is impossible.
            with _OPENER.open(req, timeout=self._timeout) as resp:
                response_bytes = resp.read(self._max_bytes)
        except HTTPError as e:
            return self._error_result(
                start, f"HTTP {e.code}: {e.reason}"
            )
        except URLError as e:
            return self._error_result(start, f"connection error: {e.reason}")
        except Exception as e:  # noqa: BLE001
            return self._error_result(
                start, f"hosted backend error: {type(e).__name__}"
            )

        try:
            parsed = json.loads(response_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return self._error_result(start, f"malformed response: {e}")

        text, finish_reason = self._extract_text_and_finish(parsed)
        if not text:
            return self._error_result(start, "empty response from backend")

        latency_ms = int((time.monotonic() - start) * 1000)

        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=int(_dig(parsed, ["usage", "prompt_tokens"]) or 0),
            completion_tokens=int(_dig(parsed, ["usage", "completion_tokens"]) or 0),
            latency_to_first_token_ms=latency_ms,
            backend=self.name,
        )

    # ----- payload builders -----

    def _openai_payload(
        self, messages: list[dict[str, Any]], params: GenerationParams
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model or "default",
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "stream": False,
        }
        if params.stop:
            payload["stop"] = list(params.stop)
        if params.seed is not None:
            payload["seed"] = params.seed
        return payload

    def _ollama_payload(
        self, messages: list[dict[str, Any]], params: GenerationParams
    ) -> dict[str, Any]:
        # Flatten chat messages into a single prompt for /api/generate
        parts: list[str] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                parts.append(f"<{role}>\n{content}\n</{role}>")
        prompt = "\n".join(parts) + "\n<assistant>\n"
        payload: dict[str, Any] = {
            "model": self._model or "default",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "num_predict": params.max_tokens,
            },
        }
        if params.stop:
            payload["options"]["stop"] = list(params.stop)
        if params.seed is not None:
            payload["options"]["seed"] = params.seed
        return payload

    # ----- response parsing -----

    @staticmethod
    def _extract_text_and_finish(parsed: dict[str, Any]) -> tuple[str, str]:
        # Try OpenAI shape first
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return (
                        msg["content"],
                        str(first.get("finish_reason") or "stop"),
                    )
                if isinstance(first.get("text"), str):
                    return (
                        first["text"],
                        str(first.get("finish_reason") or "stop"),
                    )

        # Ollama shape
        if isinstance(parsed.get("response"), str):
            done = parsed.get("done_reason") or ("stop" if parsed.get("done") else "length")
            return parsed["response"], str(done)

        # Generic text field
        if isinstance(parsed.get("text"), str):
            return parsed["text"], "stop"

        return "", "error"

    def _error_result(self, start_time: float, message: str) -> GenerationResult:
        return GenerationResult(
            text="",
            finish_reason="error",
            latency_to_first_token_ms=int((time.monotonic() - start_time) * 1000),
            backend=self.name,
            error=message,
        )


def _dig(d: dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur
