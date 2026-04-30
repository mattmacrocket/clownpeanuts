"""Deterministic stub backend.

Used for CI and offline eval. Output is computed from a hash of the
input messages so the same prompt always produces the same response.
No network, no model file, no dependencies.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from clownpeanuts.services.vuln_llm.inference.base import (
    Backend,
    GenerationParams,
    GenerationResult,
)


_STUB_PHRASES = (
    "I can help with that.",
    "Sure, here's what I know.",
    "Let me look into that for you.",
    "That's an interesting question.",
    "Based on the available context, ",
    "From what I can tell, ",
    "Here's a summary: ",
    "I'd recommend checking the documentation.",
)


class StubBackend(Backend):
    name = "stub"

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        params: GenerationParams,
    ) -> GenerationResult:
        start = time.monotonic()

        # Concatenate user content for hashing
        user_content = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    user_content += content + "\n"

        if not user_content.strip():
            user_content = "(no user content)"

        # Pick phrases deterministically based on the hash of the input
        digest = hashlib.sha256(user_content.encode("utf-8")).digest()
        seed_a = digest[0]
        seed_b = digest[1]

        phrase_a = _STUB_PHRASES[seed_a % len(_STUB_PHRASES)]
        phrase_b = _STUB_PHRASES[seed_b % len(_STUB_PHRASES)]

        # Echo a snippet of the user's question to keep responses contextual
        snippet = user_content.strip().replace("\n", " ")[:120]
        text = f"{phrase_a} {phrase_b}\n\nRegarding: \"{snippet}\""

        # Truncate to max_tokens (rough chars-per-token approximation)
        max_chars = max(8, params.max_tokens * 4)
        finish = "stop"
        if len(text) > max_chars:
            text = text[:max_chars]
            finish = "length"

        latency_ms = int((time.monotonic() - start) * 1000)

        return GenerationResult(
            text=text,
            finish_reason=finish,
            prompt_tokens=max(1, len(user_content) // 4),
            completion_tokens=max(1, len(text) // 4),
            latency_to_first_token_ms=latency_ms,
            backend=self.name,
        )
