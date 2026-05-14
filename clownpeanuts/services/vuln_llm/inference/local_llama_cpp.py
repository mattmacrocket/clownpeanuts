"""Local in-process llama.cpp inference via `llama-cpp-python`.

llama-cpp-python is an OPTIONAL dependency. The package is heavy (compiles
llama.cpp from source on install) and only relevant when you want
in-process inference. Lazy-imported so installs without the dep can still
use stub/hosted backends.

For adapter packs (manifest.model.kind == 'adapter'), the backend resolves
the base model from a model cache directory. The cache path defaults to
ClownPeanuts' standard data dir + `models/` and can be overridden via
the operator's service config.

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md §7.1.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from clownpeanuts.services.vuln_llm.inference.base import (
    Backend,
    GenerationParams,
    GenerationResult,
)


class LocalLlamaCppError(RuntimeError):
    pass


class LocalLlamaCppBackend(Backend):
    name = "local-llama-cpp"

    def __init__(
        self,
        *,
        model_path: Path,
        lora_path: Path | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = use all GPU layers (Metal/CUDA)
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as e:
            raise LocalLlamaCppError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python "
                "(macOS Metal: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python)"
            ) from e

        if not model_path.is_file():
            raise LocalLlamaCppError(f"model file not found: {model_path}")
        if lora_path is not None and not lora_path.is_file():
            raise LocalLlamaCppError(f"LoRA adapter not found: {lora_path}")

        kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        }
        if n_threads is not None:
            kwargs["n_threads"] = n_threads
        if lora_path is not None:
            kwargs["lora_path"] = str(lora_path)

        self._llm = Llama(**kwargs)
        self._model_path = model_path
        self._lora_path = lora_path

    # Bounds on per-request input shape. The emulator already caps the
    # HTTP body at 256 KiB, but within that an attacker can pack many
    # small messages or one giant `content` field. Without these caps
    # llama_cpp would spend seconds tokenizing crafted inputs, pinning
    # worker threads under modest request volume.
    _MAX_CONTENT_CHARS = 8 * 1024  # per-message
    _MAX_MESSAGES = 16             # keep last N messages

    def _truncate_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Cap each message's content length + keep only the last N
        messages. Returns a fresh list; the caller's list is not
        mutated."""
        # Take the last _MAX_MESSAGES so the most-recent context wins
        # if a malicious caller pads the front of the conversation.
        tail = messages[-self._MAX_MESSAGES:]
        out: list[dict[str, Any]] = []
        for m in tail:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if isinstance(content, str) and len(content) > self._MAX_CONTENT_CHARS:
                content = content[: self._MAX_CONTENT_CHARS]
            new_m = dict(m)
            if content is not None:
                new_m["content"] = content
            out.append(new_m)
        return out

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        params: GenerationParams,
    ) -> GenerationResult:
        start = time.monotonic()

        # Defensive truncation BEFORE handing to llama_cpp. Without
        # this, a crafted multi-MB content field forces full-context
        # tokenization (seconds of CPU) per request — easy DoS.
        messages = self._truncate_messages(messages)

        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "stream": False,
        }
        if params.stop:
            kwargs["stop"] = list(params.stop)
        if params.seed is not None:
            kwargs["seed"] = params.seed

        try:
            result: Any = self._llm.create_chat_completion(**kwargs)
        except Exception as e:  # noqa: BLE001 — llama-cpp may raise anything
            return GenerationResult(
                text="",
                finish_reason="error",
                latency_to_first_token_ms=int((time.monotonic() - start) * 1000),
                backend=self.name,
                error=f"llama_cpp error: {type(e).__name__}: {e}",
            )

        text = ""
        finish = "stop"
        if isinstance(result, dict):
            choices = result.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                if isinstance(msg.get("content"), str):
                    text = msg["content"]
                if isinstance(choices[0].get("finish_reason"), str):
                    finish = choices[0]["finish_reason"]

        usage = (result.get("usage") if isinstance(result, dict) else None) or {}

        latency_ms = int((time.monotonic() - start) * 1000)

        return GenerationResult(
            text=text,
            finish_reason=finish,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_to_first_token_ms=latency_ms,
            backend=self.name,
        )

    def close(self) -> None:
        # llama_cpp.Llama doesn't expose explicit close; let GC handle it.
        # Ensure the heavy reference is dropped so the model can be reclaimed.
        self._llm = None  # type: ignore[assignment]
