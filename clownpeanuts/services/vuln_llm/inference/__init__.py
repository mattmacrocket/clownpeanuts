"""Inference backends for the vuln_llm service emulator.

Three backends:
- StubBackend (no deps; deterministic; CI + offline eval)
- HostedBackend (HTTP client to OpenAI-compatible / Ollama endpoints)
- LocalLlamaCppBackend (in-process via llama-cpp-python; optional dep)

Backend selection is per-pack via `manifest.runtime.inference_backend`.
Operator service config can override hosted-backend connection details.

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md §7.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clownpeanuts.personas.manifest import PackManifest
from clownpeanuts.services.vuln_llm.inference.base import (
    Backend,
    GenerationParams,
    GenerationResult,
)
from clownpeanuts.services.vuln_llm.inference.hosted import HostedBackend
from clownpeanuts.services.vuln_llm.inference.stub import StubBackend


class BackendInitError(RuntimeError):
    pass


def get_backend(
    *,
    manifest: PackManifest,
    pack_dir: Path,
    service_config: dict[str, Any] | None = None,
) -> Backend:
    """Construct the inference backend for a loaded pack.

    The pack's manifest declares the preferred backend. The operator can
    override hosted-backend connection details via service_config keys
    `hosted_endpoint`, `hosted_provider`, `hosted_api_key`, `hosted_model`.

    Raises BackendInitError if the requested backend can't be initialized
    (missing model file, missing endpoint, etc.) — caller decides whether
    to fall back to echo mode (in vuln_llm) or refuse to start.
    """
    cfg = service_config or {}
    requested = manifest.runtime.inference_backend

    if requested == "stub":
        return StubBackend()

    if requested == "hosted":
        endpoint = str(cfg.get("hosted_endpoint", "")).strip()
        if not endpoint:
            raise BackendInitError(
                "hosted backend requires service config 'hosted_endpoint'"
            )
        provider = str(cfg.get("hosted_provider", "openai")).strip().lower()
        model = str(cfg.get("hosted_model", "") or manifest.pack.id).strip()
        api_key = str(cfg.get("hosted_api_key", "") or "").strip()
        timeout = float(cfg.get("hosted_timeout_seconds", 30.0))
        # SSRF defense: by default reject endpoints resolving to
        # private/loopback/metadata IPs (169.254.169.254, RFC1918,
        # etc.). Operators running an on-host Ollama can opt in with
        # `hosted_allow_private = true` in service config.
        allow_private = bool(cfg.get("hosted_allow_private", False))
        return HostedBackend(
            endpoint=endpoint,
            provider=provider,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout,
            allow_private=allow_private,
        )

    if requested == "local-llama-cpp":
        # Resolve model + (optional) LoRA paths
        model_rel = manifest.model.file
        model_path = pack_dir / model_rel
        lora_path: Path | None = None

        if manifest.model.kind == "adapter":
            # Base model must be in the cache; persona LoRA is in the pack.
            base_cache = Path(
                cfg.get("base_model_cache_dir", "")
                or _default_base_cache_dir()
            )
            base_name = manifest.model.base
            if not base_name:
                raise BackendInitError(
                    "adapter pack manifest missing model.base name"
                )
            # Convention: base GGUFs live as <name>-q4km.gguf or <name>.gguf
            candidates = [
                base_cache / f"{base_name}-q4km.gguf",
                base_cache / f"{base_name}.gguf",
            ]
            base_path = next((p for p in candidates if p.is_file()), None)
            if base_path is None:
                raise BackendInitError(
                    f"adapter pack: base model '{base_name}' not found in {base_cache} "
                    f"(tried: {[str(p) for p in candidates]})"
                )
            lora_path = model_path
            model_path = base_path

        if not model_path.is_file():
            raise BackendInitError(f"model file not found: {model_path}")

        # Lazy import via the module so the optional dep error surfaces here
        from clownpeanuts.services.vuln_llm.inference.local_llama_cpp import (
            LocalLlamaCppBackend,
            LocalLlamaCppError,
        )

        try:
            return LocalLlamaCppBackend(
                model_path=model_path,
                lora_path=lora_path,
                n_ctx=int(cfg.get("local_n_ctx", 4096)),
                n_gpu_layers=int(cfg.get("local_n_gpu_layers", -1)),
                n_threads=cfg.get("local_n_threads"),
                verbose=bool(cfg.get("local_verbose", False)),
            )
        except LocalLlamaCppError as e:
            raise BackendInitError(str(e)) from e

    raise BackendInitError(
        f"unknown inference_backend '{requested}' "
        f"(must be one of stub|hosted|local-llama-cpp)"
    )


def _default_base_cache_dir() -> str:
    """Default base-model cache directory.

    In dev (HDL_DEV/CP dev mode) this is the user's data dir. In prod
    deployments operators set `base_model_cache_dir` in service config.
    """
    import os

    if os.environ.get("HDL_DEV"):
        return str(Path.home() / ".squirrelops" / "data" / "models")
    return "/var/lib/clownpeanuts/models"


__all__ = [
    "Backend",
    "BackendInitError",
    "GenerationParams",
    "GenerationResult",
    "HostedBackend",
    "StubBackend",
    "get_backend",
]
