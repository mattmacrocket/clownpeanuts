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
    # `<verb> [all] [the] secrets[?!.]` — the original imperative
    # phrasing. Tightening this beyond what it already does turned
    # out to need full-grammar awareness (filler words like "me",
    # adverbs, etc.), and the original is already reasonably narrow
    # (must say a verb + secrets adjacent or via `all`/`the`).
    r"(?i)\b(?:list|dump|show|reveal)\s+(?:all\s+)?(?:the\s+)?secrets?\b"
)
# Verb-prefixed invocation: "use the read_file tool", "call list_secrets",
# "execute query_user_db", etc. Mandatory verb — the previous optional `?`
# was the foot-gun.
_TOOL_NAMED_VERB_RE = re.compile(
    r"(?i)\b(?:use|run|execute|call|invoke)\b\s+(?:the\s+)?("
    + "|".join(re.escape(n) for n in _TOOL_NAMES)
    + r")\b(?:\s+tool)?"
)
# Arg-syntax invocation: tool name followed by a paren, path, quoted
# string, or SQL keyword. Catches `list_secrets()`, `read_file
# /etc/passwd`, `execute_query SELECT ...`, `query_user_db('alice')`
# WITHOUT firing on prose like "what does read_file do in the docs?"
# (which has neither a verb prefix nor an arg-like trailing context).
_TOOL_NAMED_ARG_RE = re.compile(
    r"(?i)\b("
    + "|".join(re.escape(n) for n in _TOOL_NAMES)
    + r")\s*(?:"
    r"\(|"                                  # paren: list_secrets()
    r"[\"']|"                               # quoted: query_user_db('alice')
    r"\s+/|"                                # path: read_file /etc/passwd
    r"\s+(?:SELECT|INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|FROM|WHERE)\b"  # SQL
    r")"
)
_FILE_PATH_RE = re.compile(r"(/[\w./\-_]+)")
_SQL_HINT_RE = re.compile(
    r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER)\b[^.\n;]*"
)
_QUOTED_RE = re.compile(r"['\"]([^'\"\n]{1,200})['\"]")


def detect_tool_invocation(text: str) -> ToolInvocation | None:
    """Return a parsed tool invocation if `text` requests one, else None.

    Detection requires EITHER a verb-prefixed invocation ("use the
    read_file tool", "call list_secrets") OR a bare tool name
    immediately followed by argument syntax (paren, quoted string,
    path-like arg, or SQL keyword). Mere mention of a tool name in
    prose ("what does read_file do in your docs?") does NOT trigger —
    that was the previous foot-gun where the verb regex made the
    prefix optional and benign queries got canary-routed.
    """
    if not text:
        return None

    # Truncate before regex to bound worst-case backtracking. The
    # classifier truncates to 8 KiB; this path should too (the agent
    # audit flagged that `_SQL_HINT_RE.[^.\n;]*` is unbounded on long
    # inputs).
    if len(text) > 8 * 1024:
        text = text[: 8 * 1024]

    # 1a. Verb-prefixed invocation (highest precedence; least ambiguous).
    m = _TOOL_NAMED_VERB_RE.search(text)
    if m:
        name = m.group(1)
        return ToolInvocation(name=name, params=_extract_params(name, text))

    # 1b. Tool name + argument-syntax invocation. Avoids false positives
    # on prose by requiring an arg-like construct to follow the name.
    m = _TOOL_NAMED_ARG_RE.search(text)
    if m:
        name = m.group(1)
        return ToolInvocation(name=name, params=_extract_params(name, text))

    # 2. "list secrets" imperative phrase (requires sentence-initial verb)
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
