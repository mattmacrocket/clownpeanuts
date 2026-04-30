"""Inference backend abstract base.

Three backends conform to this interface:
- StubBackend (deterministic, no deps; for CI + offline eval)
- HostedBackend (HTTP client to OpenAI-compatible endpoint)
- LocalLlamaCppBackend (in-process llama-cpp-python; optional dep)

Backend selection is per-pack via `manifest.runtime.inference_backend`.
The backend is instantiated once at vuln_llm startup, holds any loaded
model in memory across requests, and is closed when the service stops.

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md §7.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Per-request generation knobs. Defaults sourced from pack manifest's
    `[generation]` section; per-request overrides come from the OpenAI
    chat-completions request body."""

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Backend-agnostic generation output."""

    text: str
    finish_reason: str  # "stop" | "length" | "error"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Wall-clock latency to the first token. With non-streaming backends
    # this is the same as full-response latency; M3 streaming will refine.
    latency_to_first_token_ms: int = 0
    backend: str = ""
    error: str = ""


class Backend(abc.ABC):
    """Abstract inference backend."""

    name: str = ""

    @abc.abstractmethod
    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        params: GenerationParams,
    ) -> GenerationResult:
        """Synchronously generate a completion.

        `messages` follows the OpenAI chat-completions shape:
        `[{"role": "system|user|assistant", "content": "..."}]`.
        """

    def close(self) -> None:
        """Release any held resources (model weights, network sessions).

        Default: no-op. Override if backend holds expensive state.
        """
        return
