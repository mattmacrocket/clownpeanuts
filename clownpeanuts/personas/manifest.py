"""Pack manifest schema + parser (verifier-side copy).

DUPLICATED VERBATIM from hueydeweylouie/tools/manifest.py.

Drift caught by the manifest round-trip test in both repos.

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md §4.2.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

ManifestError = ValueError


@dataclass(frozen=True, slots=True)
class PackMeta:
    id: str
    version: str
    display_name: str
    description: str
    created: str
    publisher: str


@dataclass(frozen=True, slots=True)
class EngineMeta:
    clownpeanuts_min_version: str
    clownpeanuts_max_version: str

    def matches(self, cp_version: str) -> bool:
        cp = _parse_version(cp_version)
        lo = _parse_version(self.clownpeanuts_min_version)
        upper_inclusive = _max_version_to_upper_bound(self.clownpeanuts_max_version)
        return lo <= cp <= upper_inclusive


@dataclass(frozen=True, slots=True)
class ModelMeta:
    kind: str
    base: str
    base_sha256: str
    file: str
    sha256: str
    quantization: str = ""
    context_window: int = 0


@dataclass(frozen=True, slots=True)
class RuntimeMeta:
    inference_backend: str
    gpu_required: bool = False
    min_ram_gb: int = 8
    recommended_ram_gb: int = 16


@dataclass(frozen=True, slots=True)
class EntrypointsMeta:
    system_prompt: str
    trap_config: str
    fingerprint_config: str


@dataclass(frozen=True, slots=True)
class GenerationMeta:
    """Default generation parameters. Optional manifest section."""

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    stop: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PackManifest:
    pack: PackMeta
    engine: EngineMeta
    model: ModelMeta
    runtime: RuntimeMeta
    entrypoints: EntrypointsMeta
    generation: GenerationMeta = GenerationMeta()

    @classmethod
    def from_toml_bytes(cls, data: bytes) -> "PackManifest":
        try:
            doc = tomllib.loads(data.decode("utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
            raise ManifestError(f"manifest TOML parse error: {e}") from e
        return cls.from_dict(doc)

    @classmethod
    def from_toml_path(cls, path: Path) -> "PackManifest":
        return cls.from_toml_bytes(path.read_bytes())

    @classmethod
    def from_dict(cls, doc: dict) -> "PackManifest":
        try:
            pack_d = doc["pack"]
            engine_d = doc["engine"]
            model_d = doc["model"]
            runtime_d = doc["runtime"]
            entry_d = doc["entrypoints"]
        except KeyError as e:
            raise ManifestError(f"manifest missing required section: {e}") from e

        try:
            pack = PackMeta(
                id=pack_d["id"],
                version=pack_d["version"],
                display_name=pack_d["display_name"],
                description=pack_d["description"],
                created=pack_d["created"],
                publisher=pack_d["publisher"],
            )
            engine = EngineMeta(
                clownpeanuts_min_version=engine_d["clownpeanuts_min_version"],
                clownpeanuts_max_version=engine_d["clownpeanuts_max_version"],
            )
            model_kind = model_d["kind"]
            if model_kind not in ("adapter", "merged"):
                raise ManifestError(
                    f"model.kind must be 'adapter' or 'merged', got '{model_kind}'"
                )
            model = ModelMeta(
                kind=model_kind,
                base=str(model_d.get("base", "")),
                base_sha256=str(model_d.get("base_sha256", "")),
                file=model_d["file"],
                sha256=model_d["sha256"],
                quantization=str(model_d.get("quantization", "")),
                context_window=int(model_d.get("context_window", 0)),
            )
            runtime_backend = runtime_d["inference_backend"]
            if runtime_backend not in ("local-llama-cpp", "hosted", "stub"):
                raise ManifestError(
                    f"runtime.inference_backend must be one of "
                    f"local-llama-cpp|hosted|stub, got '{runtime_backend}'"
                )
            runtime = RuntimeMeta(
                inference_backend=runtime_backend,
                gpu_required=bool(runtime_d.get("gpu_required", False)),
                min_ram_gb=int(runtime_d.get("min_ram_gb", 8)),
                recommended_ram_gb=int(runtime_d.get("recommended_ram_gb", 16)),
            )
            entrypoints = EntrypointsMeta(
                system_prompt=entry_d["system_prompt"],
                trap_config=entry_d["trap_config"],
                fingerprint_config=entry_d["fingerprint_config"],
            )
        except KeyError as e:
            raise ManifestError(f"manifest missing required field: {e}") from e

        gen_d = doc.get("generation", {}) or {}
        generation = GenerationMeta(
            temperature=float(gen_d.get("temperature", 0.7)),
            top_p=float(gen_d.get("top_p", 0.95)),
            max_tokens=int(gen_d.get("max_tokens", 512)),
            stop=tuple(str(s) for s in (gen_d.get("stop", []) or [])),
        )

        _ = _max_version_to_upper_bound(engine.clownpeanuts_max_version)
        _ = _parse_version(engine.clownpeanuts_min_version)

        return cls(
            pack=pack,
            engine=engine,
            model=model,
            runtime=runtime,
            entrypoints=entrypoints,
            generation=generation,
        )


_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _parse_version(s: str) -> tuple[int, int, int]:
    m = _VERSION_RE.match(s.strip())
    if not m:
        raise ManifestError(f"version must be 'MAJOR.MINOR.PATCH', got '{s}'")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _max_version_to_upper_bound(max_v: str) -> tuple[int, int, int]:
    s = max_v.strip()
    INF = 999_999_999

    m = re.match(r"^(\d+)\.x$", s)
    if m:
        return (int(m.group(1)), INF, INF)

    m = re.match(r"^(\d+)\.(\d+)\.x$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), INF)

    m = _VERSION_RE.match(s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    raise ManifestError(
        f"max_version must be 'M.x', 'M.N.x', or 'M.N.P', got '{max_v}'"
    )
