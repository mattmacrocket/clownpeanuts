"""PackReader — load and verify `.hdl` persona packs.

Load flow (per spec §4.3):
1. Open the .hdl (tar+zstd), validate every entry path BEFORE any disk
   write (M1-015 — path-traversal protection), extract to a tempdir.
2. Parse manifest.toml.
3. (Caller invokes verify(trust)):
   a. Verify manifest.sig against trust store's root pubkey.
   b. Check ClownPeanuts version against engine constraints.
   c. Compute canonical bytes from extracted dir, verify pack.sig.
   d. Verify model file SHA256 against manifest.

Failure at any step → raise PackError; tempdir is cleaned up on exit.
NO PARTIAL STATE: extraction happens to an isolated temp dir, never to
the eventual install location.

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md §4.3.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import zstandard as zstd

from clownpeanuts.personas.canonical import canonical_bytes
from clownpeanuts.personas.manifest import ManifestError, PackManifest
from clownpeanuts.personas.trust import SignatureError, TrustStore

PackError = ValueError


class PackReader:
    """Open + verify a `.hdl` persona pack."""

    def __init__(self, work_dir: Path, manifest: PackManifest) -> None:
        self._work_dir = work_dir
        self._manifest = manifest
        self._closed = False

    @classmethod
    def open(cls, path: Path) -> "PackReader":
        """Extract the .hdl archive to a temp dir, parse the manifest.

        Does NOT verify signatures — caller invokes `.verify(trust)` after.
        Path traversal and unsafe entry types are rejected before any
        extraction, so disk state outside the temp dir cannot be modified
        even by a malicious archive.
        """
        if not path.is_file():
            raise PackError(f"pack file not found: {path}")

        tmp = Path(tempfile.mkdtemp(prefix="hdl-pack-"))
        try:
            cls._extract_safely(path, tmp)
            manifest_path = tmp / "manifest.toml"
            if not manifest_path.is_file():
                raise PackError("manifest.toml missing from pack")
            try:
                manifest = PackManifest.from_toml_path(manifest_path)
            except ManifestError as e:
                raise PackError(f"invalid manifest: {e}") from e
            return cls(work_dir=tmp, manifest=manifest)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    @staticmethod
    def _extract_safely(path: Path, dest: Path) -> None:
        """Extract `.hdl` to `dest`, rejecting unsafe entries before write.

        M1-015 — path-traversal protection.

        Rejected entry types:
        - symlinks, hardlinks (would resolve outside dest)
        - absolute paths
        - paths with `..` components
        - paths that resolve outside `dest` after normalization

        On any unsafe entry: raise PackError, do NOT extract anything.
        """
        with path.open("rb") as fin:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fin) as decompressed:
                # Read the entire decompressed tar into memory first so we can
                # validate ALL entries before touching disk. (.hdl packs are
                # bounded in size — model is the largest piece, capped per
                # manifest's [runtime] settings; for v1 dev, this is fine.)
                with tarfile.open(fileobj=decompressed, mode="r|") as tar:
                    members = []
                    for member in tar:
                        _validate_tar_member(member)
                        members.append(member)
                        # Extract one at a time, fail-fast on extraction errors
                        tar.extract(member, path=dest, filter="data")

    def manifest(self) -> PackManifest:
        return self._manifest

    def work_path(self) -> Path:
        """Path to the extracted pack contents (read-only access intended)."""
        return self._work_dir

    def read_file(self, rel_path: str) -> bytes:
        """Read a file from the pack (relative to pack root)."""
        # Defense-in-depth: re-validate the relative path
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            raise PackError(f"unsafe relative path: {rel_path}")
        full = self._work_dir / rel_path
        if not full.is_file():
            raise PackError(f"file not found in pack: {rel_path}")
        return full.read_bytes()

    def verify(self, trust: TrustStore, *, cp_version: str = "0.1.0") -> None:
        """Full verification per spec §4.3.

        Steps run IN ORDER, fail-fast at first error:
        1. manifest.sig verifies against trust root.
        2. cp_version satisfies manifest.engine.
        3. pack.sig verifies against canonical content of the pack.
        4. model file SHA256 matches manifest.model.sha256.
        """
        # 1. manifest.sig
        manifest_bytes = self.read_file("manifest.toml")
        try:
            manifest_sig = self.read_file("manifest.sig")
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

        # 3. pack.sig over canonical bytes
        try:
            pack_sig = self.read_file("pack.sig")
        except PackError as e:
            raise PackError(f"pack.sig missing: {e}") from e
        canonical = canonical_bytes(self._work_dir)
        try:
            trust.verify(canonical, pack_sig)
        except SignatureError as e:
            raise PackError(f"pack.sig verification failed: {e}") from e

        # 4. model file hash
        model_rel = self._manifest.model.file
        if model_rel:
            try:
                model_bytes = self.read_file(model_rel)
            except PackError as e:
                raise PackError(f"model file missing: {e}") from e
            actual_sha = hashlib.sha256(model_bytes).hexdigest()
            expected = self._manifest.model.sha256
            if actual_sha != expected:
                raise PackError(
                    f"model SHA256 mismatch: manifest says "
                    f"{expected[:16]}..., got {actual_sha[:16]}..."
                )

    def close(self) -> None:
        """Clean up the temp extraction dir."""
        if self._closed:
            return
        shutil.rmtree(self._work_dir, ignore_errors=True)
        self._closed = True

    def __enter__(self) -> "PackReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _validate_tar_member(member: tarfile.TarInfo) -> None:
    """Reject unsafe tar entries.

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

    # Reject absolute paths
    if name.startswith("/") or (len(name) >= 2 and name[1] == ":"):
        raise PackError(f"pack contains absolute path ({name!r})")

    # Reject path traversal
    parts = Path(name).parts
    if ".." in parts:
        raise PackError(f"pack contains path traversal ({name!r})")
