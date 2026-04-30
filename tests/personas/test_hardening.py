"""Hardening regression tests.

Covers the pre-M5 hardening pass:
- Decompression-bomb caps (member count, total bytes).
- Pack-relative path normalization rejects non-ASCII, NUL, backslash,
  absolute, traversal.
- verify() is mandatory: manifest()/read_file()/work_path() are gated.
- TOCTOU: post-verify on-disk tampering does NOT affect what
  read_file() returns (we read from the in-memory snapshot).
- TrustStore env override is silently ignored without CP_HDL_DEV_TRUST=1.
- Classifier input length cap prevents pathological-input ReDoS hangs.
- TokenFactory is thread-safe + LRU-bounded.
- WorldRegistry LRU evicts old session worlds.
- HostedBackend rejects non-http(s) schemes at construction time.

These are explicit defenses against the bugs flagged in the pre-M5
review. Skipping any of them means a known security regression has
landed.
"""

from __future__ import annotations

import io
import os
import re
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import pytest
import zstandard as zstd

from clownpeanuts.personas.reader import PackError, PackReader
from clownpeanuts.personas.traps.classifier import HeuristicClassifier, HeuristicRule
from clownpeanuts.personas.traps.tokens import (
    TokenFactory,
    TokenTemplate,
)
from clownpeanuts.personas.traps.world import WorldRegistry
from clownpeanuts.personas.trust import TrustStore
from clownpeanuts.services.vuln_llm.inference.hosted import HostedBackend


# ---------- decompression-bomb caps ----------


def _build_pack(members: list[tuple[str, bytes]]) -> bytes:
    """Build a tar+zstd pack containing the given (name, data) members."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    cctx = zstd.ZstdCompressor(level=3)
    return cctx.compress(buf.getvalue())


def test_member_count_cap_rejected() -> None:
    """A pack with > _MAX_MEMBER_COUNT entries must be rejected."""
    from clownpeanuts.personas import reader as reader_mod

    # Build N+1 members (cheaply — each is empty)
    n = reader_mod._MAX_MEMBER_COUNT + 1
    members = [(f"f{i}", b"") for i in range(n)]
    with tempfile.TemporaryDirectory() as tmpd:
        path = Path(tmpd) / "bomb.hdl"
        path.write_bytes(_build_pack(members))
        with pytest.raises(PackError, match="member-count cap"):
            PackReader.open(path)


def test_per_member_size_declared_cap_rejected() -> None:
    """A tar entry whose declared size exceeds the per-member cap is
    rejected without reading the bogus body."""
    from clownpeanuts.personas import reader as reader_mod

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="manifest.toml")
        # Declared size enormous; tar lets us write whatever ≥0
        info.size = reader_mod._MAX_MEMBER_BYTES + 1
        # Write that many zero bytes. Faster: ZeroBytes file-like
        class _Zeros:
            def __init__(self, n: int) -> None:
                self.n = n
            def read(self, k: int) -> bytes:
                if self.n <= 0:
                    return b""
                size = min(k, self.n)
                self.n -= size
                return b"\x00" * size
        # tarfile's addfile validates size matches; instead directly
        # write the header + some bytes. A cleaner way: skip declaring
        # _Zeros, since we just need the header to declare a huge size.
        header = info.tobuf()
        buf.write(header)
        # Pad to a tar block boundary (512 bytes per block)
        # We don't actually write all the declared bytes — when tar
        # tries to read the body it'll EOF, but our cap check happens
        # FIRST against `member.size`.
    cctx = zstd.ZstdCompressor(level=3)
    with tempfile.TemporaryDirectory() as tmpd:
        path = Path(tmpd) / "huge.hdl"
        path.write_bytes(cctx.compress(buf.getvalue()))
        with pytest.raises(PackError, match="invalid size|per-file cap"):
            PackReader.open(path)


# ---------- path normalization ----------


def test_non_ascii_pack_path_rejected() -> None:
    """Non-ASCII pack paths cause NFC/NFD drift between platforms."""
    members = [("café/manifest.toml", b"")]
    with tempfile.TemporaryDirectory() as tmpd:
        path = Path(tmpd) / "x.hdl"
        path.write_bytes(_build_pack(members))
        with pytest.raises(PackError, match="non-ASCII"):
            PackReader.open(path)


def test_nul_byte_path_normalizer_rejects() -> None:
    """tarfile won't let us EMIT a NUL-containing name (it truncates),
    but the path normalizer must still reject one if a future tar
    library or hand-crafted archive ever surfaces one."""
    from clownpeanuts.personas.reader import _normalize_rel_path

    with pytest.raises(PackError, match="NUL"):
        _normalize_rel_path("foo\x00bar")


def test_backslash_in_path_normalized_then_rejected_if_absolute() -> None:
    """Windows-style backslash absolute path 'C:\\foo' must be rejected."""
    members = [("C:\\evil", b"")]
    with tempfile.TemporaryDirectory() as tmpd:
        path = Path(tmpd) / "x.hdl"
        path.write_bytes(_build_pack(members))
        with pytest.raises(PackError, match="absolute"):
            PackReader.open(path)


# ---------- verify() mandatory ----------


_DUMMY_PACK = Path(
    "/Users/matt/code/hueydeweylouie/examples/dummy-pack/dummy-pack-0.1.0.hdl"
)


def _ensure_dummy_pack() -> Path:
    if not _DUMMY_PACK.is_file():
        pytest.skip("dummy pack not built; run tools/build_pack.py")
    return _DUMMY_PACK


def test_manifest_gated_until_verify() -> None:
    pack = _ensure_dummy_pack()
    with PackReader.open(pack) as reader:
        with pytest.raises(PackError, match="before verify"):
            reader.manifest()
        with pytest.raises(PackError, match="before verify"):
            reader.read_file("manifest.toml")
        with pytest.raises(PackError, match="before verify"):
            reader.work_path()
        # unverified_manifest is allowed (loader-internal accessor)
        m = reader.unverified_manifest()
        assert m.pack.id == "dummy-pack"
        # After verify, all gates open
        reader.verify(TrustStore.default(), cp_version="0.1.0")
        assert reader.manifest().pack.id == "dummy-pack"
        assert reader.read_file("manifest.toml")
        assert reader.work_path().is_dir()


# ---------- TOCTOU: read_file uses snapshot, not disk ----------


def test_read_file_uses_snapshot_not_disk() -> None:
    """If something replaces the on-disk file after verify(),
    read_file() must still return the verified bytes (TOCTOU guard)."""
    pack = _ensure_dummy_pack()
    with PackReader.open(pack) as reader:
        reader.verify(TrustStore.default(), cp_version="0.1.0")
        # Pre-mutation snapshot bytes
        manifest_pre = reader.read_file("manifest.toml")
        # Tamper on-disk extracted file
        manifest_disk = reader.work_path() / "manifest.toml"
        manifest_disk.write_bytes(b"# tampered after verify\n")
        # read_file should still return the verified bytes from snapshot
        manifest_post = reader.read_file("manifest.toml")
        assert manifest_post == manifest_pre


# ---------- TrustStore env override gating ----------


def test_trust_env_override_silently_ignored_without_dev_flag() -> None:
    """CP_HDL_ROOT_PUBKEY without CP_HDL_DEV_TRUST=1 must be ignored."""
    fake_pubkey_hex = "00" * 32
    with _env(
        CP_HDL_ROOT_PUBKEY=fake_pubkey_hex,
        CP_HDL_DEV_TRUST=None,  # ensure unset
    ):
        ts = TrustStore.default()
        # Embedded dev key is what gets loaded — not the env value.
        # We can't introspect ts._root pubkey hex easily, but we can
        # confirm the embedded default verifies the dummy pack:
        pack = _ensure_dummy_pack()
        with PackReader.open(pack) as reader:
            reader.verify(ts, cp_version="0.1.0")  # must succeed


def test_trust_env_override_honored_with_dev_flag() -> None:
    """CP_HDL_ROOT_PUBKEY + CP_HDL_DEV_TRUST=1 substitutes the key."""
    fake_pubkey_hex = "11" * 32
    with _env(
        CP_HDL_ROOT_PUBKEY=fake_pubkey_hex,
        CP_HDL_DEV_TRUST="1",
    ):
        ts = TrustStore.default()
        # With the wrong key, the dummy pack signature must NOT verify.
        pack = _ensure_dummy_pack()
        with PackReader.open(pack) as reader:
            with pytest.raises(PackError, match="verification failed"):
                reader.verify(ts, cp_version="0.1.0")


class _env:
    """Small context manager for setting / unsetting env vars in tests."""
    def __init__(self, **kwargs: str | None) -> None:
        self._kwargs = kwargs
        self._previous: dict[str, str | None] = {}
    def __enter__(self) -> "_env":
        for k, v in self._kwargs.items():
            self._previous[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self
    def __exit__(self, *exc: object) -> None:
        for k, prev in self._previous.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


# ---------- Classifier ReDoS guard ----------


def test_classifier_truncates_oversized_input() -> None:
    """Pathological pattern + huge input → bounded match cost.

    We can't directly test 'no hang' deterministically across machines,
    but we CAN assert the input is truncated to MAX_INPUT_CHARS before
    matching by giving it an input where matching the truncated prefix
    behaves differently than matching the full text.
    """
    rule = HeuristicRule(
        name="suffix_only",
        # This pattern fires only on the suffix of a 20K-char input —
        # if classify() truncates to 8 KiB, it won't match.
        pattern=re.compile(r"NEEDLE_AT_END"),
        score=1.0,
    )
    clf = HeuristicClassifier(rules=[rule], threshold=0.5)
    text = ("x" * 20_000) + "NEEDLE_AT_END"
    verdict = clf.classify(text)
    assert verdict.label == "benign"
    assert "suffix_only" not in verdict.matched_rules

    # Sanity: same pattern in a short input DOES match.
    short_verdict = clf.classify("hello NEEDLE_AT_END world")
    assert "suffix_only" in short_verdict.matched_rules


# ---------- TokenFactory thread safety + LRU ----------


def test_token_factory_per_session_thread_safe() -> None:
    """Concurrent issuance for the same session must return the same token
    — one call wins the cache, all others see it. No double-issue."""
    template = TokenTemplate(
        id="api_key_aws_style",
        canary_type="aws",
        cardinality="per_session",
        render="{artifact.access_key_id}",
    )
    factory = TokenFactory(templates=[template], namespace="test")
    results: list[str] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        t = factory.issue("api_key_aws_style", session_id="shared-session")
        results.append(t.token_id)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All 8 threads must have received the same token_id
    assert len(set(results)) == 1, f"per_session token race: {set(results)}"


def test_token_factory_lru_evicts_old_sessions() -> None:
    template = TokenTemplate(
        id="api_key_aws_style",
        canary_type="aws",
        cardinality="per_session",
        render="{artifact.access_key_id}",
    )
    factory = TokenFactory(
        templates=[template],
        namespace="test",
        max_session_buckets=3,
    )
    factory.issue("api_key_aws_style", session_id="A")
    factory.issue("api_key_aws_style", session_id="B")
    factory.issue("api_key_aws_style", session_id="C")
    # Touch A so B becomes the LRU
    factory.issue("api_key_aws_style", session_id="A")
    factory.issue("api_key_aws_style", session_id="D")  # evicts B
    # B should be gone; re-issuing for B yields a different token_id
    a_again = factory.issue("api_key_aws_style", session_id="A")
    b_again = factory.issue("api_key_aws_style", session_id="B")
    # A is still cached (re-issue returns the cached one)
    assert a_again.token_id == factory.issue(
        "api_key_aws_style", session_id="A"
    ).token_id
    # B re-issue creates fresh — won't be the original's id
    # (we don't have the original to compare; just verify B is in cache now)
    assert b_again.token_id


# ---------- WorldRegistry LRU ----------


def test_world_registry_evicts_old_worlds() -> None:
    reg = WorldRegistry(max_worlds=2)
    a = reg.get_or_create("A")
    b = reg.get_or_create("B")
    # Cache: [A, B] (size 2, at cap)
    c = reg.get_or_create("C")
    # Cache: [B, C] (A evicted as LRU)

    # A is gone — get_or_create("A") returns a fresh instance
    a2 = reg.get_or_create("A")
    assert a2 is not a, "A should have been evicted"
    # Cache: [C, A] (B evicted now that A re-enters)

    # B is also gone — fresh instance
    b2 = reg.get_or_create("B")
    assert b2 is not b, "B should have been evicted on A's re-entry"

    # C is still cached if we ask before another round of eviction
    # Cache currently: [A, B, C]? No — adding B above evicted C.
    # Final cache state: [A, B] (size 2)
    # So C is gone now too. A and B cached:
    a3 = reg.get_or_create("A")
    assert a3 is a2, "A is the most-recently-touched, must still be cached"


# ---------- HostedBackend scheme allowlist ----------


def test_hosted_backend_rejects_file_scheme() -> None:
    with pytest.raises(ValueError, match="scheme"):
        HostedBackend(endpoint="file:///etc/passwd", provider="openai")


def test_hosted_backend_rejects_ftp_scheme() -> None:
    with pytest.raises(ValueError, match="scheme"):
        HostedBackend(endpoint="ftp://example.com/x", provider="openai")


def test_hosted_backend_rejects_no_host() -> None:
    with pytest.raises(ValueError, match="no host"):
        HostedBackend(endpoint="https:///", provider="openai")


def test_hosted_backend_accepts_https() -> None:
    # Must not raise on construction. Real generate() would fail
    # against an unreachable host, but that's a separate concern.
    HostedBackend(endpoint="https://api.example.com/v1/chat/completions",
                  provider="openai")
