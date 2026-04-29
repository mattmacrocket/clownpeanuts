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
    """Return the byte-exact canonical content for a pack source/extracted dir."""
    files = _collect_files(pack_dir, pack_dir)
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
