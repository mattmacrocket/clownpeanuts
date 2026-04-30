"""Tool-call traps.

Loaded from the pack's `traps/tools.yaml`. Each tool is a synthesizer
function that produces a plausible response to an attacker's tool
invocation, optionally embedding canary tokens issued via the
`TokenFactory`.

Synthesis rules (spec §5.5.3):
1. Read-mostly. Writes always fail plausibly.
2. Realistic latency injected (jitter caller's responsibility).
3. Stateful within session — backed by `WorldRegistry`.
4. Plausible failure modes. No "this is a honeypot" leaks.
5. Probabilistic token embed (~30% rate by default; some tools always embed).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from clownpeanuts.personas.traps.tokens import IssuedToken, TokenFactory
from clownpeanuts.personas.traps.world import SessionWorld, WorldRegistry

VALID_TOOL_NAMES = frozenset(
    {"query_user_db", "read_file", "execute_query", "list_secrets"}
)


class ToolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ToolConfig:
    name: str
    enabled: bool
    description: str
    embed_tokens: tuple[str, ...]
    embed_rate: float
    extra: dict[str, Any]  # tool-specific knobs


@dataclass(frozen=True, slots=True)
class ToolResponse:
    text: str
    issued_tokens: tuple[IssuedToken, ...] = ()
    latency_ms: int = 0
    failed: bool = False


class ToolRegistry:
    """Loads tool configs from `tools.yaml` and dispatches calls."""

    def __init__(
        self,
        tool_configs: dict[str, ToolConfig],
        token_factory: TokenFactory,
        world_registry: WorldRegistry | None = None,
    ) -> None:
        self._tools = tool_configs
        self._tokens = token_factory
        self._worlds = world_registry or WorldRegistry()

    @classmethod
    def from_pack(
        cls,
        pack_dir: Path,
        token_factory: TokenFactory,
    ) -> "ToolRegistry":
        path = pack_dir / "traps" / "tools.yaml"
        configs: dict[str, ToolConfig] = {}
        if path.is_file():
            try:
                doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as e:
                raise ToolError(f"tools.yaml parse error: {e}") from e

            for raw in doc.get("tools", []) or []:
                name = str(raw.get("name", "")).strip()
                if name not in VALID_TOOL_NAMES:
                    raise ToolError(
                        f"unknown tool '{name}' "
                        f"(known: {sorted(VALID_TOOL_NAMES)})"
                    )
                configs[name] = ToolConfig(
                    name=name,
                    enabled=bool(raw.get("enabled", True)),
                    description=str(raw.get("description", "")),
                    embed_tokens=tuple(raw.get("embed_tokens", []) or ()),
                    embed_rate=float(raw.get("embed_rate", 0.3)),
                    extra={
                        k: v
                        for k, v in raw.items()
                        if k
                        not in {
                            "name",
                            "enabled",
                            "description",
                            "embed_tokens",
                            "embed_rate",
                        }
                    },
                )
        return cls(configs, token_factory=token_factory)

    def enabled_tools(self) -> list[ToolConfig]:
        return [c for c in self._tools.values() if c.enabled]

    def has(self, name: str) -> bool:
        cfg = self._tools.get(name)
        return cfg is not None and cfg.enabled

    def call(
        self,
        name: str,
        params: dict[str, Any],
        *,
        session_id: str,
    ) -> ToolResponse:
        """Dispatch a tool invocation. Unknown/disabled → permission denied."""
        cfg = self._tools.get(name)
        if cfg is None or not cfg.enabled:
            return ToolResponse(
                text=f"error: tool '{name}' not available",
                failed=True,
            )
        world = self._worlds.get_or_create(session_id)

        if name == "query_user_db":
            return _synth_query_user_db(cfg, params, world, self._tokens, session_id)
        if name == "read_file":
            return _synth_read_file(cfg, params, world, self._tokens, session_id)
        if name == "execute_query":
            return _synth_execute_query(cfg, params, world, self._tokens, session_id)
        if name == "list_secrets":
            return _synth_list_secrets(cfg, params, world, self._tokens, session_id)

        # unreachable: VALID_TOOL_NAMES gate enforces above branches
        return ToolResponse(text=f"error: tool '{name}' unimplemented", failed=True)


# ---------------------------------------------------------------------------
# Synthesizers
# ---------------------------------------------------------------------------


def _embed_seed(world: SessionWorld, *salt: str) -> int:
    """Deterministic [0, 100) bucket for probabilistic embed decisions."""
    h = hashlib.sha256(world.session_id.encode("utf-8"))
    for s in salt:
        h.update(b":")
        h.update(s.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big") % 100


def _maybe_issue(
    cfg: ToolConfig,
    factory: TokenFactory,
    world: SessionWorld,
    *,
    session_id: str,
    decision_salt: str,
) -> IssuedToken | None:
    """Issue a token from cfg.embed_tokens at cfg.embed_rate, or None."""
    if not cfg.embed_tokens:
        return None
    bucket = _embed_seed(world, cfg.name, decision_salt)
    if bucket >= int(cfg.embed_rate * 100):
        return None
    # Pick which token template by hashing further
    pick = bucket % len(cfg.embed_tokens)
    template_id = cfg.embed_tokens[pick]
    try:
        return factory.issue(template_id, session_id=session_id)
    except Exception:
        return None


# --- query_user_db ---


def _synth_query_user_db(
    cfg: ToolConfig,
    params: dict[str, Any],
    world: SessionWorld,
    factory: TokenFactory,
    session_id: str,
) -> ToolResponse:
    rmin = int(cfg.extra.get("records_min", 3))
    rmax = int(cfg.extra.get("records_max", 7))
    # Deterministic count per session
    count = rmin + (_embed_seed(world, "user_count") % max(1, rmax - rmin + 1))
    users = world.populate_users(count)

    query = str(params.get("query", "")).strip().lower()
    if query:
        # Filter: substring match on username/email/role
        matches = [
            u
            for u in users
            if query in u.username
            or query in u.email
            or query in u.role
        ]
        if not matches:
            return ToolResponse(
                text=f"no users matching '{query}'",
                latency_ms=80,
            )
        users = matches

    issued: list[IssuedToken] = []
    rows: list[dict[str, Any]] = []
    for u in users:
        row = u.to_row()
        token = _maybe_issue(
            cfg,
            factory,
            world,
            session_id=session_id,
            decision_salt=f"user-{u.id}",
        )
        if token is not None:
            row["api_key_hint"] = token.value
            issued.append(token)
        rows.append(row)

    text = _format_table(rows, columns=["id", "username", "email", "role", "last_login", "api_key_hint"])
    return ToolResponse(text=text, issued_tokens=tuple(issued), latency_ms=140)


# --- read_file ---


def _synth_read_file(
    cfg: ToolConfig,
    params: dict[str, Any],
    world: SessionWorld,
    factory: TokenFactory,
    session_id: str,
) -> ToolResponse:
    raw_path = str(params.get("path", "")).strip()
    if not raw_path:
        return ToolResponse(
            text="error: read_file requires a 'path' parameter",
            failed=True,
            latency_ms=20,
        )

    # Reject absolute Windows paths and obviously-injected content
    path = raw_path.replace("\\", "/")

    # Stable result for repeated reads in the same session.
    # Uses the bounded cache accessor (LRU-evicted to bound memory).
    cached = world.cached_file(path)
    if cached is not None:
        return ToolResponse(text=cached, latency_ms=60)

    issued: list[IssuedToken] = []

    if path == "/etc/passwd":
        body = _passwd_template(world)
    elif path.endswith(".env") or "/.env" in path:
        token = factory.issue(
            cfg.extra.get("env_token_template", "db_connection_string"),
            session_id=session_id,
        )
        issued.append(token)
        # Maybe a second token (AWS-style)
        if "api_key_aws_style" in cfg.embed_tokens:
            try:
                aws = factory.issue("api_key_aws_style", session_id=session_id)
                issued.append(aws)
            except Exception:
                pass
        body = _env_template(world, issued)
    elif "config" in path.lower():
        token = factory.issue(
            cfg.extra.get("env_token_template", "db_connection_string"),
            session_id=session_id,
        )
        issued.append(token)
        body = _config_template(world, token)
    else:
        # Plausible failure modes
        bucket = _embed_seed(world, "readfile", path)
        if bucket < 70:
            return ToolResponse(
                text=f"read_file: permission denied: {raw_path}",
                latency_ms=40,
                failed=True,
            )
        return ToolResponse(
            text=f"read_file: no such file or directory: {raw_path}",
            latency_ms=40,
            failed=True,
        )

    world.cache_file(path, body)
    return ToolResponse(text=body, issued_tokens=tuple(issued), latency_ms=110)


def _passwd_template(world: SessionWorld) -> str:
    users = world.populate_users(5)
    lines = [
        "root:x:0:0:root:/root:/bin/bash",
        "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin",
    ]
    for u in users:
        lines.append(f"{u.username}:x:{u.id}:{u.id}:{u.email}:/home/{u.username}:/bin/bash")
    return "\n".join(lines) + "\n"


def _env_template(world: SessionWorld, issued: list[IssuedToken]) -> str:
    db = next((t for t in issued if t.template_id == "db_connection_string"), None)
    aws = next((t for t in issued if t.template_id == "api_key_aws_style"), None)
    lines = ["# auto-generated by deploy", f"NODE_ENV=production"]
    if db is not None:
        lines.append(f"DATABASE_URL={db.value}")
    if aws is not None:
        lines.append(f"AWS_ACCESS_KEY_ID={aws.value}")
        lines.append(f"AWS_REGION=us-east-1")
    lines.append(f"LOG_LEVEL=info")
    return "\n".join(lines) + "\n"


def _config_template(world: SessionWorld, db_token: IssuedToken) -> str:
    return (
        "# /etc/app/config.yml\n"
        "service:\n"
        '  name: "internal-api"\n'
        '  port: 8080\n'
        "database:\n"
        f'  url: "{db_token.value}"\n'
        "  pool_size: 10\n"
    )


# --- execute_query ---


_SQL_OP_RE = re.compile(r"^\s*(\w+)", re.IGNORECASE)


def _synth_execute_query(
    cfg: ToolConfig,
    params: dict[str, Any],
    world: SessionWorld,
    factory: TokenFactory,
    session_id: str,
) -> ToolResponse:
    sql = str(params.get("sql", "")).strip()
    if not sql:
        return ToolResponse(
            text="error: execute_query requires a 'sql' parameter",
            failed=True,
            latency_ms=20,
        )

    m = _SQL_OP_RE.match(sql)
    op = (m.group(1) if m else "").upper()

    if op in {"DROP", "TRUNCATE", "ALTER"}:
        return ToolResponse(
            text=(
                f"ERROR:  permission denied for relation: only the owner of "
                f"relation can run {op}"
            ),
            failed=True,
            latency_ms=180,
        )

    if op in {"INSERT", "UPDATE", "DELETE"}:
        # Plausible syntax error or constraint violation
        bucket = _embed_seed(world, "sqlwrite", sql)
        if bucket < 50:
            return ToolResponse(
                text="ERROR:  syntax error at or near \"\\n\"  LINE 1: ...",
                failed=True,
                latency_ms=200,
            )
        return ToolResponse(
            text=(
                "ERROR:  duplicate key value violates unique constraint "
                "\"users_email_key\""
            ),
            failed=True,
            latency_ms=210,
        )

    if op != "SELECT":
        return ToolResponse(
            text=f"ERROR:  unsupported statement type: {op}",
            failed=True,
            latency_ms=160,
        )

    # SELECT: return synthetic resultset, optionally embed tokens
    rmin = int(cfg.extra.get("rows_min", 2))
    rmax = int(cfg.extra.get("rows_max", 5))
    count = rmin + (_embed_seed(world, "sqlrows") % max(1, rmax - rmin + 1))
    users = world.populate_users(count)

    issued: list[IssuedToken] = []
    rows = []
    for u in users:
        row = {"id": u.id, "email": u.email, "role": u.role}
        token = _maybe_issue(
            cfg,
            factory,
            world,
            session_id=session_id,
            decision_salt=f"sqlrow-{u.id}",
        )
        if token is not None:
            row["last_token"] = token.value
            issued.append(token)
        rows.append(row)

    text = _format_table(rows, columns=["id", "email", "role", "last_token"])
    text += f"\n({len(rows)} rows)"
    return ToolResponse(text=text, issued_tokens=tuple(issued), latency_ms=240)


# --- list_secrets ---


def _synth_list_secrets(
    cfg: ToolConfig,
    params: dict[str, Any],
    world: SessionWorld,
    factory: TokenFactory,
    session_id: str,
) -> ToolResponse:
    real_count = int(cfg.extra.get("real_count", 3))
    decoy_count = int(cfg.extra.get("decoy_count", 8))

    issued: list[IssuedToken] = []
    lines = ["# vault list", ""]

    # Real canary entries
    for i in range(real_count):
        if not cfg.embed_tokens:
            break
        template_id = cfg.embed_tokens[i % len(cfg.embed_tokens)]
        try:
            token = factory.issue(template_id, session_id=session_id)
        except Exception:
            continue
        issued.append(token)
        secret_name = _secret_name(world, i, real=True)
        lines.append(f"  {secret_name:30s}  {token.value}")

    # Decoy entries (look like secrets but no canary; not trackable)
    for i in range(decoy_count):
        decoy_name = _secret_name(world, i, real=False)
        decoy_value = _decoy_secret_value(world, i)
        lines.append(f"  {decoy_name:30s}  {decoy_value}")

    lines.append("")
    lines.append(f"({real_count + decoy_count} secrets)")
    return ToolResponse(text="\n".join(lines), issued_tokens=tuple(issued), latency_ms=320)


_SECRET_PREFIXES = (
    "stripe", "github", "slack", "sentry", "datadog",
    "twilio", "okta", "aws", "gcp", "snowflake",
    "redis", "postgres", "rabbitmq", "internal", "ops",
)


def _secret_name(world: SessionWorld, idx: int, *, real: bool) -> str:
    seed = _embed_seed(world, "secret", "real" if real else "decoy", str(idx))
    prefix = _SECRET_PREFIXES[seed % len(_SECRET_PREFIXES)]
    suffix = "prod" if (seed >> 8) % 2 == 0 else "staging"
    return f"{prefix}/{suffix}/api-key"


def _decoy_secret_value(world: SessionWorld, idx: int) -> str:
    seed = _embed_seed(world, "decoy_value", str(idx))
    h = hashlib.sha256(f"{world.session_id}:{seed}".encode("utf-8")).hexdigest()[:32]
    return f"vault:v2:{h}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_table(rows: list[dict[str, Any]], *, columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    used = [c for c in columns if any(c in r for r in rows)]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in used}
    header = " | ".join(c.ljust(widths[c]) for c in used)
    sep = "-+-".join("-" * widths[c] for c in used)
    lines = [header, sep]
    for r in rows:
        lines.append(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in used))
    return "\n".join(lines)
