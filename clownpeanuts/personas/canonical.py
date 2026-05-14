"""Canonical pack content serializer for `pack.sig` verification.

DUPLICATED VERBATIM from hueydeweylouie/tools/canonical.py — see that file
for design rationale (independence from json library version, byte-identical
output across implementations).

Drift between the two copies is caught by the canonical-bytes regression
test pinning the SHA256 of canonical bytes for a fixed input. Update both
files together, or the verification will silently break.

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md §4.4.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

CANONICAL_SCHEMA = "hdl-pack-v1"
EXCLUDED_PATHS: frozenset[str] = frozenset({"pack.sig"})


def canonical_bytes(pack_dir: Path) -> bytes:
    """Return the byte-exact canonical content for a pack source/extracted dir.

    Reads files from disk. Used by the writer (build_pack.py) and by
    legacy callers. The verifier path uses
    `canonical_bytes_from_snapshot()` instead so verification is
    independent of post-extract disk state (TOCTOU guard).
    """
    files = _collect_files(pack_dir, pack_dir)
    return _serialize_canonical(files)


def canonical_bytes_from_snapshot(snapshot: dict[str, bytes]) -> bytes:
    """Return the byte-exact canonical content from an in-memory snapshot.

    `snapshot` keys are forward-slash relative paths (as produced by
    PackReader._load_snapshot). `pack.sig` is excluded; everything else
    contributes one `{"path": "...", "sha256": "..."}` entry sorted
    lexicographically by path.

    This is the verifier-side entry point. The writer uses
    `canonical_bytes(pack_dir)` (disk-walking) since the writer owns the
    bytes anyway and TOCTOU is moot before signing.
    """
    files: list[tuple[str, bytes]] = []
    for rel_path, content in snapshot.items():
        if rel_path in EXCLUDED_PATHS:
            continue
        files.append((rel_path, content))
    return _serialize_canonical(files)


def canonical_bytes_from_hashes(file_hashes: dict[str, str]) -> bytes:
    """Same canonical-bytes output as `canonical_bytes_from_snapshot`,
    but takes pre-computed SHA-256 hex hashes per file rather than
    raw bytes.

    Defends against the decompression-bomb RAM issue: a multi-GB
    model file no longer needs to be held in memory to compute the
    pack signature. The PackReader streams large members directly
    to disk + computes the hash inline; this function consumes that
    hash map.

    Output bytes are byte-identical to `canonical_bytes_from_snapshot`
    given equivalent inputs, so signatures verify against either.
    """
    files = [
        (rel, sha)
        for rel, sha in file_hashes.items()
        if rel not in EXCLUDED_PATHS
    ]
    files.sort(key=lambda item: item[0])
    parts: list[str] = ['{"files":[']
    for i, (rel_path, sha) in enumerate(files):
        if i > 0:
            parts.append(",")
        parts.append('{"path":')
        parts.append(_json_string(rel_path))
        parts.append(',"sha256":"')
        parts.append(sha)
        parts.append('"}')
    parts.append('],"schema":"')
    parts.append(CANONICAL_SCHEMA)
    parts.append('"}')
    return "".join(parts).encode("utf-8")


def _serialize_canonical(files: list[tuple[str, bytes]]) -> bytes:
    files.sort(key=lambda item: item[0])

    parts: list[str] = ['{"files":[']
    for i, (rel_path, content) in enumerate(files):
        if i > 0:
            parts.append(",")
        digest = hashlib.sha256(content).hexdigest()
        parts.append('{"path":')
        parts.append(_json_string(rel_path))
        parts.append(',"sha256":"')
        parts.append(digest)
        parts.append('"}')
    parts.append('],"schema":"')
    parts.append(CANONICAL_SCHEMA)
    parts.append('"}')

    return "".join(parts).encode("utf-8")


def _collect_files(directory: Path, base: Path) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    for entry in os.scandir(directory):
        if entry.is_dir(follow_symlinks=False):
            sub = Path(entry.path)
            out.extend(_collect_files(sub, base))
        elif entry.is_file(follow_symlinks=False):
            full = Path(entry.path)
            rel = full.relative_to(base)
            rel_str = "/".join(rel.parts)
            if rel_str in EXCLUDED_PATHS:
                continue
            out.append((rel_str, full.read_bytes()))
    return out


def _json_string(s: str) -> str:
    chunks: list[str] = ['"']
    for ch in s:
        codepoint = ord(ch)
        if ch == '"':
            chunks.append("\\\"")
        elif ch == "\\":
            chunks.append("\\\\")
        elif ch == "\n":
            chunks.append("\\n")
        elif ch == "\r":
            chunks.append("\\r")
        elif ch == "\t":
            chunks.append("\\t")
        elif ch == "\b":
            chunks.append("\\b")
        elif ch == "\f":
            chunks.append("\\f")
        elif codepoint < 0x20:
            chunks.append(f"\\u{codepoint:04x}")
        else:
            chunks.append(ch)
    chunks.append('"')
    return "".join(chunks)
