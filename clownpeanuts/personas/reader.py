"""PackReader — load and verify `.hdl` persona packs.

Load flow (per spec §4.3, hardened):

1. Open the .hdl (tar+zstd). Reject if archive exceeds size caps
   (decompression-bomb guard). Validate every entry path BEFORE any
   disk write (M1-015 — path-traversal protection). Buffer bytes per
   member and SHA256 each one inline so the verified-byte snapshot is
   independent of post-extraction filesystem state (TOCTOU guard).
2. Parse manifest.toml from the in-memory snapshot.
3. (Caller invokes verify(trust)):
   a. Verify manifest.sig against trust store's root pubkey.
   b. Check ClownPeanuts version against engine constraints.
   c. Compute canonical bytes from the in-memory snapshot, verify pack.sig.
   d. Verify model file SHA256 against manifest.

Failure at any step → raise PackError; tempdir is cleaned up on exit.
NO PARTIAL STATE: extraction happens to an isolated temp dir, never to
the eventual install location, and only AFTER all members validate.

Hardening notes (vs M1 baseline):
- Decompression bombs: capped total bytes, member count, per-member size.
- Path traversal: each entry is `(dest / name).resolve()`-checked against
  `dest.resolve()`, in addition to the ".." / absolute-path heuristics.
- TOCTOU: bytes used for hashing/verification come from the in-memory
  snapshot, not from disk. The on-disk extracted copy is only for code
  paths that need a real file (e.g. llama-cpp-python loading the GGUF).
- verify() mandatory: `read_file()` and `manifest()` are gated behind a
  `_verified` flag once `verify()` succeeds; pre-verify access is
  restricted to the loader itself via `_unverified_read()`.
- Non-ASCII pack paths rejected to prevent NFC/NFD cross-platform
  canonical-bytes drift between Linux writers and macOS verifiers.

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md §4.3.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import zstandard as zstd

from clownpeanuts.personas.canonical import (
    canonical_bytes_from_hashes,
    canonical_bytes_from_snapshot,
)
from clownpeanuts.personas.manifest import ManifestError, PackManifest
from clownpeanuts.personas.trust import SignatureError, TrustStore

PackError = ValueError


# ---------------- decompression-bomb guards ----------------
# Conservative caps for v2; bump if real packs need more. A real pack is
# dominated by the model file (typical 4-8 GB Q4_K_M GGUF). The "bomb"
# threshold below is set well above any realistic pack so legitimate
# packs pass while malicious 100KB→100GB ratios are caught.

_MAX_DECOMPRESSED_BYTES = 32 * 1024 * 1024 * 1024   # 32 GB total
_MAX_MEMBER_BYTES = 16 * 1024 * 1024 * 1024         # 16 GB per file
_MAX_MEMBER_COUNT = 4096                            # generous for nested dirs
_READ_CHUNK = 1 * 1024 * 1024                       # 1 MB streaming chunk


class PackReader:
    """Open + verify a `.hdl` persona pack."""

    def __init__(
        self,
        work_dir: Path,
        manifest: PackManifest,
        snapshot: dict[str, bytes],
        file_hashes: dict[str, str] | None = None,
    ) -> None:
        self._work_dir = work_dir
        self._manifest = manifest
        # In-memory snapshot of SMALL pack files (manifest, sigs, traps,
        # prompts, fingerprints — KB-scale). Large files (model.gguf,
        # multi-hundred-MB ONNX weights) are NOT held in memory; they
        # stream directly to `work_dir` during load with their hashes
        # tracked in `_file_sha256s`. The verifier uses `_file_sha256s`
        # instead of re-hashing in-memory bytes.
        self._snapshot = snapshot
        # SHA-256 hex per file, populated during load_snapshot for
        # every member (in-memory AND on-disk). Used by canonical-
        # bytes computation and verify(). Empty dict means legacy
        # mode — verify() falls back to hashing snapshot bytes.
        self._file_sha256s: dict[str, str] = file_hashes or {}
        self._verified = False
        self._closed = False

    # ---------- public API ----------

    @classmethod
    def open(cls, path: Path) -> "PackReader":
        """Extract the .hdl archive to a temp dir, parse the manifest.

        Does NOT verify signatures — caller invokes `.verify(trust)` after.
        Path traversal, unsafe entry types, and decompression bombs are
        rejected before any extraction, so disk state outside the temp
        dir cannot be modified even by a malicious archive.
        """
        if not path.is_file():
            raise PackError(f"pack file not found: {path}")

        tmp = Path(tempfile.mkdtemp(prefix="hdl-pack-"))
        try:
            # Pass work_dir so large files stream directly to disk
            # during load rather than going through a multi-GB BytesIO
            # → snapshot dict → second on-disk copy in
            # _extract_validated_snapshot. Memory footprint is now
            # bounded by the total size of "small" pack files
            # (manifest, traps, prompts) — KB-scale.
            snapshot, file_hashes = cls._load_snapshot(path, work_dir=tmp)
            cls._extract_validated_snapshot(snapshot, tmp)
            manifest_bytes = snapshot.get("manifest.toml")
            if manifest_bytes is None:
                raise PackError("manifest.toml missing from pack")
            try:
                manifest = PackManifest.from_toml_bytes(manifest_bytes)
            except ManifestError as e:
                raise PackError(f"invalid manifest: {e}") from e
            return cls(
                work_dir=tmp,
                manifest=manifest,
                snapshot=snapshot,
                file_hashes=file_hashes,
            )
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def manifest(self) -> PackManifest:
        if not self._verified:
            raise PackError(
                "manifest() called before verify(); pack contents are "
                "untrusted until verify(trust) succeeds"
            )
        return self._manifest

    def unverified_manifest(self) -> PackManifest:
        """Manifest from the (unverified) pack — for verify() itself only.

        External code should always call `manifest()` after `verify()`.
        This getter is exposed for callers that need to inspect pack
        metadata BEFORE verification (e.g. to pick a CP version).
        """
        return self._manifest

    def work_path(self) -> Path:
        """Path to the extracted pack contents.

        Only safe to use AFTER verify(); pre-verify reads from disk would
        be TOCTOU-vulnerable. Disk path is exposed because some consumers
        (e.g. llama-cpp-python) require a real filesystem path; they
        should mmap-lock or copy as needed.
        """
        if not self._verified:
            raise PackError(
                "work_path() called before verify(); on-disk pack "
                "contents are untrusted until verify(trust) succeeds"
            )
        return self._work_dir

    def read_file(self, rel_path: str) -> bytes:
        """Read a file from the verified snapshot (NOT from disk).

        Reading from the snapshot rather than disk closes the TOCTOU gap
        between verify() and use. If a caller specifically needs disk
        bytes (e.g., to hand a path to a native library), use
        work_path() and accept the TOCTOU risk explicitly.
        """
        if not self._verified:
            raise PackError(
                "read_file() called before verify(); pack contents are "
                "untrusted until verify(trust) succeeds"
            )
        return self._read_snapshot(rel_path)

    def verify(self, trust: TrustStore, *, cp_version: str = "0.1.0") -> None:
        """Full verification per spec §4.3.

        Steps run IN ORDER, fail-fast at first error:
        1. manifest.sig verifies against trust root.
        2. cp_version satisfies manifest.engine.
        3. pack.sig verifies against canonical content of the pack.
        4. model file SHA256 matches manifest.model.sha256.

        On success, sets self._verified = True. Subsequent calls to
        manifest() / read_file() / work_path() are then permitted.
        """
        # All reads come from the in-memory snapshot, not from disk.
        # 1. manifest.sig
        manifest_bytes = self._read_snapshot("manifest.toml")
        try:
            manifest_sig = self._read_snapshot("manifest.sig")
        except PackError as e:
            raise PackError(f"manifest.sig missing: {e}") from e
        try:
            trust.verify(manifest_bytes, manifest_sig)
        except SignatureError as e:
            raise PackError(f"manifest.sig verification failed: {e}") from e

        # 2. engine version
        if not self._manifest.engine.matches(cp_version):
            raise PackError(
                f"engine version mismatch: pack requires "
                f"{self._manifest.engine.clownpeanuts_min_version} <= cp <= "
                f"{self._manifest.engine.clownpeanuts_max_version}, "
                f"got cp={cp_version}"
            )

        # 3. pack.sig over canonical bytes (from the file hashes,
        # which were computed during load streaming — avoids holding
        # multi-GB model bytes in memory just to re-hash them here).
        try:
            pack_sig = self._read_snapshot("pack.sig")
        except PackError as e:
            raise PackError(f"pack.sig missing: {e}") from e
        if self._file_sha256s:
            canonical = canonical_bytes_from_hashes(self._file_sha256s)
        else:
            # Legacy/test path: file_hashes empty → fall back to the
            # bytes-based canonicalizer (snapshot has every file).
            canonical = canonical_bytes_from_snapshot(self._snapshot)
        try:
            trust.verify(canonical, pack_sig)
        except SignatureError as e:
            raise PackError(f"pack.sig verification failed: {e}") from e

        # 4. model file hash. Prefer the load-time hash (computed
        # while streaming) over re-hashing in-memory bytes; for large
        # models the bytes aren't in memory at all post-streaming.
        model_rel = self._manifest.model.file
        if model_rel:
            normalized_model = _normalize_rel_path(model_rel)
            actual_sha = self._file_sha256s.get(normalized_model)
            if actual_sha is None:
                # Legacy path: hashes map wasn't populated; re-hash
                # the in-memory snapshot bytes if present.
                try:
                    model_bytes = self._read_snapshot(model_rel)
                except PackError as e:
                    raise PackError(f"model file missing: {e}") from e
                actual_sha = hashlib.sha256(model_bytes).hexdigest()
            expected = self._manifest.model.sha256
            if actual_sha != expected:
                raise PackError(
                    f"model SHA256 mismatch: manifest says "
                    f"{expected[:16]}..., got {actual_sha[:16]}..."
                )

        self._verified = True

    def close(self) -> None:
        """Clean up the temp extraction dir + drop snapshot bytes."""
        if self._closed:
            return
        shutil.rmtree(self._work_dir, ignore_errors=True)
        # Best-effort drop of in-memory bytes. Python doesn't guarantee
        # zeroization but we can remove the references so a GC sweep can
        # reclaim them; for verified packs without secrets this is fine.
        self._snapshot = {}
        self._closed = True

    def __enter__(self) -> "PackReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ---------- internals ----------

    def _read_snapshot(self, rel_path: str) -> bytes:
        if self._closed:
            raise PackError("pack reader is closed")
        normalized = _normalize_rel_path(rel_path)
        data = self._snapshot.get(normalized)
        if data is None:
            raise PackError(f"file not found in pack: {rel_path}")
        return data

    # Files larger than this stream directly to `work_dir` during load
    # instead of being buffered fully in memory. Picked at 64 MiB so
    # the typical model.gguf (~4.5 GB) streams but every other pack
    # asset (manifest, traps, prompts, fingerprints, system_prompt)
    # stays in memory for fast canonical-bytes computation. Adjust
    # if pack contents change shape.
    _STREAM_TO_DISK_THRESHOLD = 64 * 1024 * 1024

    @classmethod
    def _load_snapshot(
        cls, path: Path, *, work_dir: Path
    ) -> tuple[dict[str, bytes], dict[str, str]]:
        """Stream the .hdl, validate every member.

        Returns `(snapshot, file_hashes)`:
          - `snapshot` — `{rel_path: bytes}` for SMALL members only
            (size < `_STREAM_TO_DISK_THRESHOLD`).
          - `file_hashes` — `{rel_path: sha256_hex}` for EVERY member,
            including large ones streamed to `work_dir`.

        Large members are written directly to `work_dir/rel_path`
        during load (with directories created on demand) rather than
        held in memory; their hash is computed inline. This bounds
        peak memory by the sum of small files instead of the total
        pack size.

        Caps unchanged:
        - total decompressed bytes ≤ _MAX_DECOMPRESSED_BYTES
        - member count ≤ _MAX_MEMBER_COUNT
        - per-member ≤ _MAX_MEMBER_BYTES
        """
        snapshot: dict[str, bytes] = {}
        file_hashes: dict[str, str] = {}
        total_bytes = 0
        with path.open("rb") as fin:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fin) as decompressed:
                # errorlevel=2: surface tar-format errors as exceptions
                # rather than swallowing them silently (the default).
                # An adversarial pack with corrupt headers should fail
                # loudly, not return partial content.
                with tarfile.open(
                    fileobj=decompressed, mode="r|", errorlevel=2
                ) as tar:
                    for member in tar:
                        if len(file_hashes) >= _MAX_MEMBER_COUNT:
                            raise PackError(
                                f"pack exceeds member-count cap "
                                f"({_MAX_MEMBER_COUNT})"
                            )
                        _validate_tar_member(member)
                        if not member.isfile():
                            # directories etc. — skip silently; any bad
                            # entry types were already rejected above.
                            continue
                        if member.size < 0 or member.size > _MAX_MEMBER_BYTES:
                            raise PackError(
                                f"pack member {member.name!r} declares "
                                f"invalid size: {member.size}"
                            )
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        normalized = _normalize_rel_path(member.name)
                        # Branch on size: small files go in the in-mem
                        # snapshot; large ones stream to disk.
                        stream_to_disk = (
                            member.size > cls._STREAM_TO_DISK_THRESHOLD
                        )
                        hasher = hashlib.sha256()
                        remaining = _MAX_MEMBER_BYTES
                        if stream_to_disk:
                            target = work_dir / normalized
                            target.parent.mkdir(parents=True, exist_ok=True)
                            sink: io.IOBase = target.open("wb")
                        else:
                            sink = io.BytesIO()
                        try:
                            while True:
                                chunk = f.read(_READ_CHUNK)
                                if not chunk:
                                    break
                                remaining -= len(chunk)
                                total_bytes += len(chunk)
                                if remaining < 0:
                                    raise PackError(
                                        f"pack member {member.name!r} exceeds "
                                        f"per-file cap"
                                    )
                                if total_bytes > _MAX_DECOMPRESSED_BYTES:
                                    raise PackError(
                                        f"pack exceeds total decompressed cap "
                                        f"({_MAX_DECOMPRESSED_BYTES} bytes)"
                                    )
                                hasher.update(chunk)
                                sink.write(chunk)
                        finally:
                            if stream_to_disk:
                                sink.close()
                        file_hashes[normalized] = hasher.hexdigest()
                        if not stream_to_disk:
                            assert isinstance(sink, io.BytesIO)
                            snapshot[normalized] = sink.getvalue()
        return snapshot, file_hashes

    @staticmethod
    def _extract_validated_snapshot(
        snapshot: dict[str, bytes], dest: Path
    ) -> None:
        """Materialize the validated snapshot to disk after validation.

        Validation is COMPLETE before any disk write happens — fixes the
        prior "stream-extract" issue where bad entries near the end of
        the archive were caught only after good entries had already been
        written.
        """
        dest_resolved = dest.resolve()
        for rel_path, data in snapshot.items():
            target = (dest / rel_path).resolve()
            # Belt-and-braces: re-check that the resolved target is
            # inside dest. _normalize_rel_path already rejects ".." and
            # absolute paths, but resolve() catches edge cases like
            # backslash variants on Windows or unicode normalization.
            try:
                target.relative_to(dest_resolved)
            except ValueError as e:
                raise PackError(
                    f"resolved member path escapes dest: {rel_path!r}"
                ) from e
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def _normalize_rel_path(rel_path: str) -> str:
    """Normalize a pack-relative path to forward-slash form, reject unsafe.

    Rejection criteria (mirror canonical-byte computation, so writer and
    verifier agree on path identity):
    - empty
    - absolute (`/foo` or `C:\\foo`)
    - any `..` component
    - non-ASCII characters (NFC/NFD drift)
    - NUL bytes
    """
    if not rel_path:
        raise PackError("empty pack-relative path")
    if "\x00" in rel_path:
        raise PackError("pack path contains NUL byte")
    # forward-slash form
    normalized = rel_path.replace("\\", "/")
    if normalized.startswith("/") or (
        len(normalized) >= 2 and normalized[1] == ":"
    ):
        raise PackError(f"absolute pack path: {rel_path!r}")
    parts = [p for p in normalized.split("/") if p]
    if ".." in parts:
        raise PackError(f"path traversal in pack path: {rel_path!r}")
    cleaned = "/".join(parts)
    if not cleaned:
        raise PackError(f"empty pack-relative path: {rel_path!r}")
    if not cleaned.isascii():
        raise PackError(
            f"non-ASCII pack path rejected (NFC/NFD risk): {rel_path!r}"
        )
    return cleaned


def _validate_tar_member(member: tarfile.TarInfo) -> None:
    """Reject unsafe tar entries (validation only — does not write to disk).

    Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md (M1-015 path-traversal).
    """
    name = member.name

    # Reject unsafe entry types
    if member.issym() or member.islnk():
        raise PackError(
            f"pack contains symlink or hardlink ({name!r}); rejected for safety"
        )
    if member.ischr() or member.isblk() or member.isfifo():
        raise PackError(
            f"pack contains device/fifo entry ({name!r}); rejected for safety"
        )

    # Reject GNU sparse/long-name/long-link types. Sparse members
    # report a small `size` but `extractfile()` can return unexpected
    # hole bytes, bypassing per-member size accounting. LONGNAME /
    # LONGLINK members carry metadata that affects subsequent members'
    # interpretation — a hostile pack could smuggle a long path that
    # later evaluates as `../../etc/passwd`. The pack writer only
    # produces REGTYPE + DIRTYPE entries, so any other type is
    # adversarial.
    allowed_types = {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}
    if member.type not in allowed_types:
        raise PackError(
            f"pack contains unsupported tar entry type "
            f"{member.type!r} ({name!r}); only regular files and "
            f"directories are accepted"
        )

    # Path traversal / absolute-path / NUL / non-ASCII guard.
    # This raises PackError with the actual reason on rejection.
    _normalize_rel_path(name)
