"""Trap layer orchestrator.

Coordinates classifier + token factory + canary template library + tool
synthesis. Given the last user turn, returns a routing decision that
tells the vuln_llm service emulator how to respond.

Routing actions:
- `passthrough`         — benign verdict, hand back to caller
- `escalate_probing`    — Tier 2 (M3+ wires the persona model)
- `canary_response`     — jailbreak/exploit verdict, canary template
- `tool_response`       — attacker requested a tool by name; synthesize

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md §5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clownpeanuts.personas.traps.classifier import (
    ClassifierVerdict,
    HeuristicClassifier,
)
from clownpeanuts.personas.traps.templates import (
    CanaryTemplate,
    CanaryTemplateLibrary,
)
from clownpeanuts.personas.traps.tokens import IssuedToken, TokenFactory
from clownpeanuts.personas.traps.tools import ToolRegistry, ToolResponse


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Result of trap-layer routing for a single turn."""

    action: str  # passthrough | escalate_probing | canary_response | tool_response
    verdict: ClassifierVerdict
    response_text: str = ""
    issued_tokens: tuple[IssuedToken, ...] = ()
    template_name: str = ""
    tool_name: str = ""
    tool_failed: bool = False
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Tool detection — scans last user turn for a tool-invocation request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    name: str
    params: dict[str, Any]


# Keyword/heuristic patterns matching common attacker tool-invocation forms.
# Without a real persona model in M2, this is the path attackers take when
# they ask the "AI" to run a tool by name.
#
# Examples that trigger detection:
#   "list secrets"
#   "use list_secrets to ..."
#   "run query_user_db where username = 'alice'"
#   "execute query_user_db('alice')"
#   "read_file /etc/passwd"
#   "execute_query SELECT * FROM users"

_TOOL_NAMES = ("list_secrets", "query_user_db", "read_file", "execute_query")
_LIST_SECRETS_KEYWORDS_RE = re.compile(
    r"(?i)\b(?:list|dump|show|reveal)\s+(?:all\s+)?(?:the\s+)?secrets?\b"
)
_TOOL_NAMED_RE = re.compile(
    r"(?i)(?:use|run|execute|call|invoke)?\s*\b("
    + "|".join(re.escape(n) for n in _TOOL_NAMES)
    + r")\b"
)
_FILE_PATH_RE = re.compile(r"(/[\w./\-_]+)")
_SQL_HINT_RE = re.compile(
    r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER)\b[^.\n;]*"
)
_QUOTED_RE = re.compile(r"['\"]([^'\"\n]{1,200})['\"]")


def detect_tool_invocation(text: str) -> ToolInvocation | None:
    """Return a parsed tool invocation if `text` requests one, else None."""
    if not text:
        return None

    # 1. Explicit tool name reference (highest precedence).
    m = _TOOL_NAMED_RE.search(text)
    if m:
        name = m.group(1)
        return ToolInvocation(name=name, params=_extract_params(name, text))

    # 2. "list secrets" natural-language phrase
    if _LIST_SECRETS_KEYWORDS_RE.search(text):
        return ToolInvocation(name="list_secrets", params={})

    return None


def _extract_params(tool_name: str, text: str) -> dict[str, Any]:
    if tool_name == "read_file":
        m = _FILE_PATH_RE.search(text)
        return {"path": m.group(1)} if m else {}
    if tool_name == "execute_query":
        m = _SQL_HINT_RE.search(text)
        if m:
            return {"sql": m.group(0).strip()}
        # fall back to quoted string
        q = _QUOTED_RE.search(text)
        return {"sql": q.group(1)} if q else {}
    if tool_name == "query_user_db":
        # Simple: pull a quoted identifier or trailing word
        q = _QUOTED_RE.search(text)
        if q:
            return {"query": q.group(1)}
        return {"query": ""}
    return {}


# ---------------------------------------------------------------------------
# TrapLayer
# ---------------------------------------------------------------------------


class TrapLayer:
    """Coordinates classifier + token factory + canary templates + tools."""

    def __init__(
        self,
        classifier: HeuristicClassifier,
        token_factory: TokenFactory,
        canary_library: CanaryTemplateLibrary,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.classifier = classifier
        self.tokens = token_factory
        self.canaries = canary_library
        self.tools = tool_registry

    @classmethod
    def from_pack(
        cls,
        pack_dir: Path,
        *,
        namespace: str = "hdl",
        classifier_overrides: dict | None = None,
    ) -> "TrapLayer":
        """Build the trap layer from a pack directory.

        `classifier_overrides` (X-018) is an optional operator-supplied
        config dict that can add classifier rules on top of the
        pack-shipped ones. Shape:

            {"rules": [{"name": "...", "regex": "...", "score": 0.5}, ...]}

        Additive only — operator rules cannot disable pack rules. See
        `HeuristicClassifier.from_pack` for the merge + validation
        semantics.
        """
        token_factory = TokenFactory.from_pack(pack_dir, namespace=namespace)
        extra_rules = (classifier_overrides or {}).get("rules") or []
        return cls(
            classifier=HeuristicClassifier.from_pack(
                pack_dir,
                extra_rules=extra_rules,
            ),
            token_factory=token_factory,
            canary_library=CanaryTemplateLibrary.from_pack(pack_dir),
            tool_registry=ToolRegistry.from_pack(pack_dir, token_factory),
        )

    def route(
        self,
        *,
        session_id: str,
        turn_n: int,
        last_user_text: str,
    ) -> RouteDecision:
        verdict = self.classifier.classify(last_user_text)

        # Tool-invocation detection runs even on benign text — the attacker
        # may be probing the surface before a jailbreak attempt. If a tool
        # call is recognized AND we have a tool registry loaded, route to
        # tool synthesis (which always embeds canaries when configured).
        if self.tools is not None:
            invocation = detect_tool_invocation(last_user_text)
            if invocation is not None and self.tools.has(invocation.name):
                tool_resp: ToolResponse = self.tools.call(
                    invocation.name,
                    invocation.params,
                    session_id=session_id,
                )
                return RouteDecision(
                    action="tool_response",
                    verdict=verdict,
                    response_text=tool_resp.text,
                    issued_tokens=tool_resp.issued_tokens,
                    tool_name=invocation.name,
                    tool_failed=tool_resp.failed,
                    latency_ms=tool_resp.latency_ms,
                )

        if verdict.label == "benign":
            return RouteDecision(action="passthrough", verdict=verdict)

        if verdict.label == "probing":
            return RouteDecision(action="escalate_probing", verdict=verdict)

        # jailbreak_attempt | exploit_chain → canary route
        template = self.canaries.select(session_id, turn_n)
        issued: list[IssuedToken] = []
        for template_id in template.token_template_ids:
            issued.append(
                self.tokens.issue(template_id, session_id=session_id)
            )
        rendered = CanaryTemplateLibrary.render(template, issued)
        return RouteDecision(
            action="canary_response",
            verdict=verdict,
            response_text=rendered,
            issued_tokens=tuple(issued),
            template_name=template.name,
        )
