"""TrustStore — Ed25519 signature verification + revocation list.

The SquirrelOps root pubkey is embedded in this module at source level.
For dev builds, the embedded key is the dev-root key from
`hueydeweylouie/examples/dev-keys/dev-root.pub.hex`. Production builds
substitute the real root pubkey at release time.

In dev/test runs, the embedded key can be overridden via the
`CP_HDL_ROOT_PUBKEY` env var (64-char hex). Production releases should
NOT honor this override — gate it on a separate dev flag if you keep it.

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md §8.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


# Dev-key fallback. Replace at production release time with the real
# SquirrelOps root pubkey hex. (For now, this matches
# hueydeweylouie/examples/dev-keys/dev-root.pub.hex.)
_EMBEDDED_DEV_ROOT_PUBKEY_HEX = (
    "e69f78452df66b4bbf236179c8c2c6ffdc0c59d32e89df63ece8cb1b8a311901"
)


SignatureError = ValueError
RevocationError = ValueError


class TrustStore:
    """Holds the trusted root pubkey + revocation list."""

    def __init__(
        self,
        root_pubkey: Ed25519PublicKey,
        revoked_pubkeys: list[bytes] | None = None,
    ) -> None:
        self._root = root_pubkey
        self._revoked: list[bytes] = list(revoked_pubkeys or [])

    @classmethod
    def default(cls) -> "TrustStore":
        """Construct with the embedded root pubkey (dev override allowed)."""
        env_override = os.environ.get("CP_HDL_ROOT_PUBKEY")
        hex_str = (env_override or _EMBEDDED_DEV_ROOT_PUBKEY_HEX).strip()
        if len(hex_str) != 64:
            raise SignatureError(
                f"root pubkey hex must be 64 chars, got {len(hex_str)}"
            )
        try:
            raw = bytes.fromhex(hex_str)
        except ValueError as e:
            raise SignatureError(f"invalid root pubkey hex: {e}") from e
        return cls(Ed25519PublicKey.from_public_bytes(raw))

    @property
    def root_pubkey(self) -> Ed25519PublicKey:
        return self._root

    def add_revocation(self, pubkey_bytes: bytes) -> None:
        """Add a 32-byte raw pubkey to the revocation list."""
        if len(pubkey_bytes) != 32:
            raise RevocationError(
                f"revocation pubkey must be 32 bytes, got {len(pubkey_bytes)}"
            )
        self._revoked.append(bytes(pubkey_bytes))

    def is_revoked(self, pubkey: Ed25519PublicKey) -> bool:
        from cryptography.hazmat.primitives import serialization

        raw = pubkey.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return raw in self._revoked

    def verify(self, message: bytes, signature: bytes) -> None:
        """Verify `signature` over `message` against the root pubkey.

        Raises SignatureError on bad sig, revoked key, or wrong sig length.
        """
        if self.is_revoked(self._root):
            raise SignatureError("root key has been revoked")
        if len(signature) != 64:
            raise SignatureError(
                f"Ed25519 signature must be 64 bytes, got {len(signature)}"
            )
        try:
            self._root.verify(signature, message)
        except InvalidSignature as e:
            raise SignatureError(f"signature verification failed: {e}") from e

    def load_revocations(self, path: Path) -> None:
        """Load revoked pubkeys from a JSON file.

        Schema: `{"schema":"hdl-revocation-v1","revoked":["<hex>", ...]}`
        Missing file → no-op (no revocations is the default state).
        Malformed schema → hard error (don't silently proceed with stale data).
        """
        if not path.exists():
            return

        try:
            doc = json.loads(path.read_bytes().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise RevocationError(
                f"revocation list at {path} is malformed: {e}"
            ) from e

        schema = doc.get("schema")
        if schema != "hdl-revocation-v1":
            raise RevocationError(
                f"revocation list at {path}: unsupported schema '{schema}' "
                f"(expected 'hdl-revocation-v1')"
            )

        revoked = doc.get("revoked", [])
        if not isinstance(revoked, list):
            raise RevocationError(
                f"revocation list at {path}: 'revoked' must be a list"
            )

        for hex_str in revoked:
            if not isinstance(hex_str, str) or len(hex_str) != 64:
                raise RevocationError(
                    f"revocation list at {path}: each entry must be 64-char hex"
                )
            try:
                raw = bytes.fromhex(hex_str)
            except ValueError as e:
                raise RevocationError(
                    f"revocation list at {path}: invalid hex '{hex_str}': {e}"
                ) from e
            self._revoked.append(raw)
