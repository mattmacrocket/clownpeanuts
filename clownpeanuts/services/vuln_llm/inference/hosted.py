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

import ipaddress
import json
import socket
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


def _is_private_or_special_ip(addr_str: str) -> bool:
    """Return True if `addr_str` is a private, loopback, link-local,
    multicast, or otherwise reserved IP address.

    Defends against SSRF via operator-config tampering:
    - 127.0.0.1 / ::1 (loopback)
    - 169.254.169.254 (AWS/Azure/GCP IMDS metadata service)
    - 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16 (RFC1918 private)
    - 100.64.0.0/10 (CGNAT — internal corporate ranges)
    - fc00::/7 (IPv6 ULA)
    """
    try:
        ip = ipaddress.ip_address(addr_str)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_hosted_endpoint(endpoint: str, *, allow_private: bool = False) -> None:
    """Reject SSRF-shaped hosted endpoints.

    Beyond the existing scheme check, this rejects:
    1. URLs with userinfo (`http://google.com@evil.example/`) — the
       `netloc` containing `@` parses as `userinfo@host`, and the host
       actually resolved is the post-@ part. Easy SSRF if an attacker
       can tamper with operator config.
    2. URLs with query or fragment components — operator should not
       be putting auth tokens or other secrets in the URL itself,
       and trailing query strings often leak into upstream logs.
    3. Hostnames that resolve (via getaddrinfo) to private,
       loopback, link-local, multicast, or reserved IPs unless
       `allow_private` is explicitly set. Most importantly, this
       blocks the AWS/Azure/GCP instance metadata service at
       169.254.169.254 which can yield instance credentials.
    4. Literal IP-address hosts that are private/special.

    Raises ValueError with a clear message on rejection.
    """
    parsed = urlparse(endpoint)

    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            f"HostedBackend: endpoint has userinfo (username/password "
            f"in URL); reject because `http://google.com@evil.example/` "
            f"shape is an SSRF vector. Put credentials in the "
            f"Authorization header via `hosted_api_key_from`."
        )
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"HostedBackend: endpoint must not contain a query or "
            f"fragment (got {endpoint!r}); put authentication in the "
            f"Authorization header, not the URL."
        )

    host = parsed.hostname
    if not host:
        raise ValueError(
            f"HostedBackend: endpoint has no host: {endpoint!r}"
        )

    # Reject IDN homoglyphs / non-ASCII hosts up front.
    try:
        host.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(
            f"HostedBackend: non-ASCII hostname {host!r} rejected "
            f"(IDN homoglyph SSRF risk); use the explicit punycode "
            f"form if this is intentional."
        ) from None

    if allow_private:
        return

    # If host is a literal IP, check directly. Otherwise resolve via
    # getaddrinfo and check every result — a hostname might resolve
    # to BOTH a public and a private IP (split-horizon DNS).
    if _is_private_or_special_ip(host):
        raise ValueError(
            f"HostedBackend: endpoint host {host!r} is a "
            f"private/loopback/reserved IP; set `hosted_allow_private "
            f"= true` in service config if this is intentional "
            f"(e.g. on-host Ollama)."
        )
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        # If resolution fails at config time, fall through — the actual
        # request will fail with a clear connection error rather than
        # us hard-failing at service load on DNS hiccups.
        return
    for info in infos:
        sockaddr = info[4]
        if sockaddr and _is_private_or_special_ip(str(sockaddr[0])):
            raise ValueError(
                f"HostedBackend: endpoint host {host!r} resolves to "
                f"private/reserved IP {sockaddr[0]!r}; set "
                f"`hosted_allow_private = true` in service config if "
                f"this is intentional (e.g. on-host Ollama)."
            )


def _sanitize_error(s: str, *, max_len: int = 200) -> str:
    """Truncate and strip control characters from an error message.

    Upstream HTTP error reasons can include the URL (which may have
    contained query-string credentials before our validator rejected
    those configs) or bytes from the response body. Bound the length
    and strip control chars to prevent log spoofing / token leakage.
    """
    if not isinstance(s, str):
        s = repr(s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    if len(s) > max_len:
        s = s[:max_len] + "...(truncated)"
    return s


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
        allow_private: bool = False,
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
        # Full SSRF defense: userinfo, query/fragment, IDN, private IPs.
        # `allow_private=True` is an explicit operator opt-in for the
        # legitimate on-host-Ollama deployment shape.
        _validate_hosted_endpoint(endpoint, allow_private=allow_private)
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
            # Upstream `reason` can include attacker-controlled bytes
            # (a compromised upstream sets the reason phrase to anything).
            # Sanitize before logging.
            return self._error_result(
                start, f"HTTP {e.code}: {_sanitize_error(str(e.reason))}"
            )
        except URLError as e:
            return self._error_result(
                start,
                f"connection error: {_sanitize_error(str(e.reason))}",
            )
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
