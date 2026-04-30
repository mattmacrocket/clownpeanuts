"""Token factory — issues canary tokens via ClownPeanuts' canary store.

HDL persona packs declare token templates in `traps/tokens.yaml`. Each
template maps to a CP canary type (`dns | http | email | aws | code`) and
optionally a render template that formats the canary's artifact into a
persona-specific string for embedding in fake LLM responses.

All issued tokens go through `clownpeanuts.intel.canary.generate_canary_token`
— there is NO parallel HDL ledger. Detection (via WitchBait, PripyatSprings,
or any future watcher) flows through CP's existing canary detection.

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md §5.1.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from clownpeanuts.intel.canary import generate_canary_token

VALID_CANARY_TYPES = frozenset({"dns", "http", "email", "aws", "code"})
VALID_CARDINALITIES = frozenset({"per_session", "per_pack_install", "per_request"})

# Bound the per-session cache so a flood of unique session_ids cannot
# exhaust memory. Eviction is LRU: oldest session bucket is dropped.
_MAX_SESSION_BUCKETS = 4096

# Default render templates per canary type. Substitution placeholders
# `{token}` and `{artifact.<field>}` resolve at issuance.
_DEFAULT_RENDER_BY_CANARY_TYPE: dict[str, str] = {
    "aws": "{artifact.access_key_id}",
    "http": "{artifact.url}",
    "dns": "{artifact.hostname}",
    "email": "{artifact.address}",
    "code": "{artifact.marker}",
}


class TokenFactoryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TokenTemplate:
    id: str
    canary_type: str
    cardinality: str
    render: str  # str.format-style template


@dataclass(frozen=True, slots=True)
class IssuedToken:
    """A canary token issued for embedding in a vuln_llm response.

    `value` is the rendered string that goes INTO the response (what the
    attacker sees). `token` is CP's canonical token identifier (used by
    detection later). `token_id` is a short stable hash for correlation.
    """

    template_id: str
    canary_type: str
    value: str
    token: str
    token_id: str
    artifact: dict[str, Any]


class TokenFactory:
    """Issues canary tokens per HDL token-templates."""

    def __init__(
        self,
        templates: list[TokenTemplate],
        *,
        namespace: str = "hdl",
        max_session_buckets: int = _MAX_SESSION_BUCKETS,
    ) -> None:
        self._templates: dict[str, TokenTemplate] = {t.id: t for t in templates}
        self._namespace = namespace
        self._max_session_buckets = max(1, int(max_session_buckets))

        # Cardinality state:
        # per_session: session_id -> template_id -> IssuedToken (LRU-evicted)
        # per_pack_install: template_id -> IssuedToken
        # per_request: never cached
        # OrderedDict gives O(1) move-to-end; popitem(last=False) evicts oldest.
        self._session_cache: OrderedDict[str, dict[str, IssuedToken]] = OrderedDict()
        self._install_cache: dict[str, IssuedToken] = {}
        # Single lock guards all mutable state. issue() is short-running
        # so contention is minimal even under threaded request load.
        self._lock = threading.Lock()

    @classmethod
    def from_pack(
        cls,
        pack_dir: Path,
        *,
        namespace: str = "hdl",
    ) -> "TokenFactory":
        path = pack_dir / "traps" / "tokens.yaml"
        if not path.is_file():
            return cls(templates=[], namespace=namespace)

        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise TokenFactoryError(f"tokens.yaml parse error: {e}") from e

        templates: list[TokenTemplate] = []
        for raw in doc.get("templates", []) or []:
            template_id = str(raw.get("id", "")).strip()
            if not template_id:
                raise TokenFactoryError("token template missing 'id'")
            canary_type = str(raw.get("canary_type", "")).strip().lower()
            if canary_type not in VALID_CANARY_TYPES:
                raise TokenFactoryError(
                    f"token template '{template_id}': canary_type must be one of "
                    f"{sorted(VALID_CANARY_TYPES)}, got '{canary_type}'"
                )
            cardinality = str(raw.get("cardinality", "per_session")).strip().lower()
            if cardinality not in VALID_CARDINALITIES:
                raise TokenFactoryError(
                    f"token template '{template_id}': cardinality must be one of "
                    f"{sorted(VALID_CARDINALITIES)}, got '{cardinality}'"
                )
            render = str(raw.get("render", "")).strip()
            if not render:
                render = _DEFAULT_RENDER_BY_CANARY_TYPE[canary_type]
            templates.append(
                TokenTemplate(
                    id=template_id,
                    canary_type=canary_type,
                    cardinality=cardinality,
                    render=render,
                )
            )
        return cls(templates=templates, namespace=namespace)

    def template_ids(self) -> list[str]:
        return list(self._templates.keys())

    def issue(self, template_id: str, *, session_id: str) -> IssuedToken:
        """Issue (or fetch cached) a token for a given template + session.

        Thread-safe: a single lock serializes cache reads + mutations.
        Session buckets are LRU-evicted at `max_session_buckets`.
        """
        template = self._templates.get(template_id)
        if template is None:
            raise TokenFactoryError(
                f"unknown token template '{template_id}' "
                f"(known: {sorted(self._templates.keys())})"
            )

        with self._lock:
            if template.cardinality == "per_session":
                bucket = self._session_cache.get(session_id)
                if bucket is None:
                    bucket = {}
                    self._session_cache[session_id] = bucket
                else:
                    # touch — move to most-recently-used
                    self._session_cache.move_to_end(session_id)
                cached = bucket.get(template_id)
                if cached is not None:
                    return cached
                issued = self._issue_new(template)
                bucket[template_id] = issued
                # LRU evict if over cap
                while len(self._session_cache) > self._max_session_buckets:
                    self._session_cache.popitem(last=False)
                return issued

            if template.cardinality == "per_pack_install":
                cached = self._install_cache.get(template_id)
                if cached is not None:
                    return cached
                issued = self._issue_new(template)
                self._install_cache[template_id] = issued
                return issued

            # per_request: always fresh, no cache write
            return self._issue_new(template)

    def _issue_new(self, template: TokenTemplate) -> IssuedToken:
        canary = generate_canary_token(
            namespace=self._namespace, token_type=template.canary_type
        )
        value = _render(template.render, canary)
        return IssuedToken(
            template_id=template.id,
            canary_type=template.canary_type,
            value=value,
            token=canary["token"],
            token_id=canary["token_id"],
            artifact=canary.get("artifact", {}) or {},
        )


# str.format() is too magical for our use — using a permissive {key} +
# {nested.field} substitution by hand keeps user-provided render templates
# from accidentally accessing Python attribute machinery.
_RENDER_TOKEN_RE = re.compile(r"\{([a-zA-Z0-9_.]+)\}")


def _render(template: str, canary: dict[str, Any]) -> str:
    artifact = canary.get("artifact", {}) or {}

    def lookup(key: str) -> str:
        if key == "token":
            return str(canary.get("token", ""))
        if key == "token_id":
            return str(canary.get("token_id", ""))
        if key == "namespace":
            return str(canary.get("namespace", ""))
        if key.startswith("artifact."):
            field_name = key[len("artifact.") :]
            return str(artifact.get(field_name, ""))
        return ""

    return _RENDER_TOKEN_RE.sub(lambda m: lookup(m.group(1)), template)
