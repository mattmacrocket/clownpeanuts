"""End-to-end tests for the persona-pack format.

Covers M1's eval gate:
- Round-trip: build-then-verify the dummy pack.
- Tamper-manifest: modified manifest.toml → manifest.sig fails.
- Tamper-pack-sig: modified pack.sig → pack.sig verification fails.
- Tamper-model: modified model file → pack.sig fails (canonical includes hash).
- Version-mismatch: cp_version outside manifest engine range → reject.
- Path-traversal: malicious .hdl with `..` entries → reject before extraction.
- Canonical-bytes regression: pinned SHA256 for fixed input.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pytest
import zstandard as zstd

from clownpeanuts.personas.canonical import canonical_bytes
from clownpeanuts.personas.manifest import ManifestError, PackManifest
from clownpeanuts.personas.reader import PackError, PackReader
from clownpeanuts.personas.trust import TrustStore

HDL_REPO = Path("/Users/matt/code/hueydeweylouie")
DUMMY_PACK_HDL = HDL_REPO / "examples" / "dummy-pack" / "dummy-pack-0.1.0.hdl"


def _ensure_dummy_pack_built() -> Path:
    """Build the v2 dummy pack via tools/build_pack.py if not present."""
    if DUMMY_PACK_HDL.is_file():
        return DUMMY_PACK_HDL
    pytest.skip(
        "dummy-pack-0.1.0.hdl not built. Run: "
        f"python {HDL_REPO}/tools/build_pack.py {HDL_REPO}/examples/dummy-pack"
    )
    return DUMMY_PACK_HDL


# ---------- M1-008: round-trip ----------


def test_round_trip_open_and_verify() -> None:
    pack = _ensure_dummy_pack_built()
    with PackReader.open(pack) as reader:
        # Pre-verify: only unverified_manifest() is allowed.
        assert reader.unverified_manifest().pack.id == "dummy-pack"
        assert reader.unverified_manifest().pack.version == "0.1.0"
        # Pre-verify: manifest() and read_file() are gated.
        with pytest.raises(PackError, match="before verify"):
            reader.manifest()
        with pytest.raises(PackError, match="before verify"):
            reader.read_file("manifest.toml")
        # cp_version 0.1.0 is in the dummy pack's range (0.1.0 .. 0.x)
        reader.verify(TrustStore.default(), cp_version="0.1.0")
        # Post-verify: now manifest() and read_file() work.
        assert reader.manifest().pack.id == "dummy-pack"
        assert reader.read_file("manifest.toml")


# ---------- M1-009: tamper manifest ----------


def test_tamper_manifest_fails_sig() -> None:
    pack = _ensure_dummy_pack_built()

    with tempfile.TemporaryDirectory() as tmpd:
        tampered = _tamper_pack(pack, Path(tmpd), tamper_path="manifest.toml",
                                tamper_fn=lambda b: b + b"\n# trailing garbage\n")
        with PackReader.open(tampered) as reader, pytest.raises(
            PackError, match="manifest.sig"
        ):
            reader.verify(TrustStore.default(), cp_version="0.1.0")


# ---------- M1-010: tamper pack.sig ----------


def test_tamper_pack_sig_fails() -> None:
    pack = _ensure_dummy_pack_built()

    def flip_first_byte(b: bytes) -> bytes:
        return bytes([b[0] ^ 0x01]) + b[1:]

    with tempfile.TemporaryDirectory() as tmpd:
        tampered = _tamper_pack(pack, Path(tmpd), tamper_path="pack.sig",
                                tamper_fn=flip_first_byte)
        with PackReader.open(tampered) as reader, pytest.raises(
            PackError, match="pack.sig"
        ):
            reader.verify(TrustStore.default(), cp_version="0.1.0")


# ---------- M1-011: tamper model ----------


def test_tamper_model_fails_pack_sig() -> None:
    pack = _ensure_dummy_pack_built()

    with tempfile.TemporaryDirectory() as tmpd:
        tampered = _tamper_pack(pack, Path(tmpd), tamper_path="model/model.gguf",
                                tamper_fn=lambda b: b + b"EXTRA")
        with PackReader.open(tampered) as reader, pytest.raises(
            PackError, match="pack.sig"  # canonical bytes change → pack.sig fails first
        ):
            reader.verify(TrustStore.default(), cp_version="0.1.0")


# ---------- M1-012: version mismatch ----------


def test_version_mismatch_rejected() -> None:
    pack = _ensure_dummy_pack_built()
    # Dummy pack range: cp_min=0.1.0, cp_max=0.x → reject 1.0.0
    with PackReader.open(pack) as reader, pytest.raises(
        PackError, match="engine version mismatch"
    ):
        reader.verify(TrustStore.default(), cp_version="1.0.0")


# ---------- M1-013: bad max_version (manifest-level) ----------


def test_bad_max_version_rejected_at_parse() -> None:
    bad = b"""
[pack]
id = "x"
version = "1.0.0"
display_name = "x"
description = "x"
created = "2026-01-01T00:00:00Z"
publisher = "x"

[engine]
clownpeanuts_min_version = "0.1.0"
clownpeanuts_max_version = "garbage"

[model]
kind = "merged"
file = "model/m.gguf"
sha256 = "abc"

[runtime]
inference_backend = "stub"

[entrypoints]
system_prompt = "p.md"
trap_config = "t.yaml"
fingerprint_config = "f.yaml"
"""
    with pytest.raises(ManifestError, match="max_version"):
        PackManifest.from_toml_bytes(bad)


# ---------- M1-014: canonical bytes regression ----------


def test_canonical_bytes_pinned_for_known_input() -> None:
    """Must produce the SAME hash as hueydeweylouie/tools/tests/test_canonical.py.

    If this drifts, the writer and verifier will disagree → all packs break.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "manifest.toml").write_bytes(b"[pack]\nid = \"test\"\n")
        (d / "manifest.sig").write_bytes(b"\x00" * 64)
        (d / "model").mkdir()
        (d / "model" / "model.gguf").write_bytes(b"DUMMY_GGUF")
        (d / "pack.sig").write_bytes(b"\xff" * 64)

        bytes_out = canonical_bytes(d)
        digest = hashlib.sha256(bytes_out).hexdigest()

        # Must match the writer-side pin in
        # hueydeweylouie/tools/tests/test_canonical.py
        EXPECTED = "64f82ffcb8c7b6dba70dd7876250af2a774121bf8ef2928322ce165240fd50a5"
        assert digest == EXPECTED, (
            f"canonical bytes drift detected!\n"
            f"  writer side (hueydeweylouie) pins: {EXPECTED}\n"
            f"  verifier side (clownpeanuts) got:  {digest}\n"
            f"  bytes:    {bytes_out!r}"
        )


# ---------- M1-015: path-traversal rejection ----------


def test_path_traversal_rejected() -> None:
    """Construct a malicious .hdl with `../escape.txt` and verify it fails BEFORE writing."""
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)

        # Build a tar with a path-traversal entry
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            data = b"escape!"
            info = tarfile.TarInfo(name="../escape.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        # Compress with zstd
        cctx = zstd.ZstdCompressor(level=3)
        bad_pack = tmp / "bad.hdl"
        bad_pack.write_bytes(cctx.compress(buf.getvalue()))

        canary = tmp / "escape.txt"
        with pytest.raises(PackError, match="path traversal"):
            PackReader.open(bad_pack)

        # And — the canary file outside the temp must not have been written
        assert not (tmp.parent / "escape.txt").exists()
        assert not canary.exists()


def test_symlink_entry_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name="link.txt")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        cctx = zstd.ZstdCompressor(level=3)
        bad_pack = tmp / "bad.hdl"
        bad_pack.write_bytes(cctx.compress(buf.getvalue()))

        with pytest.raises(PackError, match="symlink"):
            PackReader.open(bad_pack)


# ---------- helpers ----------


def _tamper_pack(
    src: Path,
    work_dir: Path,
    *,
    tamper_path: str,
    tamper_fn,
) -> Path:
    """Extract `src`, mutate `tamper_path` via `tamper_fn`, repack to `work_dir/tampered.hdl`.

    The repacked file does NOT re-sign — that's the point: we want the
    verifier to detect the tampering.
    """
    extract_dir = work_dir / "extracted"
    extract_dir.mkdir()

    # Extract
    with src.open("rb") as fin:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fin) as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|") as tar:
                tar.extractall(path=extract_dir, filter="data")

    # Mutate
    target = extract_dir / tamper_path
    target.write_bytes(tamper_fn(target.read_bytes()))

    # Repack (deterministic)
    files = sorted(extract_dir.rglob("*"))
    cctx = zstd.ZstdCompressor(level=3)
    out_path = work_dir / "tampered.hdl"
    with out_path.open("wb") as fout:
        with cctx.stream_writer(fout, closefd=False) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                for p in files:
                    if not p.is_file():
                        continue
                    rel = p.relative_to(extract_dir)
                    rel_posix = "/".join(rel.parts)
                    info = tar.gettarinfo(name=str(p), arcname=rel_posix)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mode = 0o644
                    with p.open("rb") as f:
                        tar.addfile(info, fileobj=f)
    return out_path
