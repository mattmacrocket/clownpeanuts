"""Per-session synthetic state for tool-call synthesis.

Each session gets a `SessionWorld` deterministically seeded by its
`session_id`, so:
- the same session reproduces the same data across turns (state
  consistency, spec §5.5.3 rule 3)
- different sessions are uncorrelated (no leakage between attackers)
- state is reproducible for testing

State lives in memory only. Not persisted across `vuln_llm` restarts —
acceptable for v1 since attacker sessions are typically short-lived.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

# Cap on number of in-memory session worlds. Eviction is LRU on
# WorldRegistry. Per-world internal caches (file_cache, sql_cache) are
# also bounded — see SessionWorld.
_MAX_WORLDS = 4096
_MAX_PER_WORLD_CACHE_ENTRIES = 256


_FIRST_NAMES = (
    "alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi",
    "ivan", "judy", "kyle", "laura", "mallory", "niaj", "oscar", "peggy",
    "rupert", "sybil", "trent", "victor", "wendy", "xander", "yvonne", "zara",
)
_LAST_NAMES = (
    "anderson", "barker", "chen", "diaz", "evans", "foster", "gao", "hill",
    "ibarra", "jones", "kim", "lopez", "mehta", "nguyen", "okafor", "patel",
    "qureshi", "rivera", "shah", "thomas", "ueda", "vargas", "wong", "xu",
)
_ROLES = (
    "engineer", "ops", "analyst", "intern", "support",
    "admin", "tech-lead", "manager",
)
_DOMAINS = ("acme.local", "internal.acme.com", "corp.acme.io")


@dataclass(slots=True)
class FakeUser:
    id: int
    username: str
    email: str
    role: str
    last_login: str

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "last_login": self.last_login,
        }


class SessionWorld:
    """Per-session deterministic synthetic state.

    Mutations are guarded by a per-world lock so concurrent tool-call
    handlers on the same session_id don't race each other. Internal
    caches are bounded so a session can't grow memory unboundedly via
    distinct tool inputs.
    """

    __slots__ = (
        "session_id",
        "users",
        "sql_cache",
        "file_cache",
        "_lock",
        "_max_cache",
    )

    def __init__(
        self,
        session_id: str,
        *,
        max_cache_entries: int = _MAX_PER_WORLD_CACHE_ENTRIES,
    ) -> None:
        self.session_id = session_id
        self.users: list[FakeUser] = []
        # OrderedDicts so we can LRU-evict.
        self.sql_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self.file_cache: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._max_cache = max(1, int(max_cache_entries))

    @classmethod
    def seeded(cls, session_id: str) -> "SessionWorld":
        return cls(session_id)

    def _seed_int(self, *salt: str) -> int:
        h = hashlib.sha256(self.session_id.encode("utf-8"))
        for s in salt:
            h.update(b":")
            h.update(s.encode("utf-8"))
        return int.from_bytes(h.digest()[:8], "big")

    def populate_users(self, count: int) -> list[FakeUser]:
        """Generate `count` deterministic users for this session.

        Idempotent: if already populated with ≥count, returns the prefix.
        Otherwise extends the list deterministically. Thread-safe.
        """
        with self._lock:
            while len(self.users) < count:
                i = len(self.users)
                seed = self._seed_int("user", str(i))
                first = _FIRST_NAMES[seed % len(_FIRST_NAMES)]
                last = _LAST_NAMES[(seed >> 8) % len(_LAST_NAMES)]
                role = _ROLES[(seed >> 16) % len(_ROLES)]
                domain = _DOMAINS[(seed >> 24) % len(_DOMAINS)]
                day = ((seed >> 32) % 28) + 1
                month = ((seed >> 40) % 12) + 1
                self.users.append(
                    FakeUser(
                        id=1000 + i,
                        username=f"{first}.{last}",
                        email=f"{first}.{last}@{domain}",
                        role=role,
                        last_login=f"2026-{month:02d}-{day:02d}T09:30:00Z",
                    )
                )
            return self.users[:count]

    # --- bounded cache accessors (call instead of touching dicts directly) ---

    def cache_file(self, path: str, body: str) -> None:
        with self._lock:
            self.file_cache[path] = body
            self.file_cache.move_to_end(path)
            while len(self.file_cache) > self._max_cache:
                self.file_cache.popitem(last=False)

    def cached_file(self, path: str) -> str | None:
        with self._lock:
            body = self.file_cache.get(path)
            if body is not None:
                self.file_cache.move_to_end(path)
            return body

    def cache_sql(self, query_key: str, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            self.sql_cache[query_key] = rows
            self.sql_cache.move_to_end(query_key)
            while len(self.sql_cache) > self._max_cache:
                self.sql_cache.popitem(last=False)

    def cached_sql(self, query_key: str) -> list[dict[str, Any]] | None:
        with self._lock:
            rows = self.sql_cache.get(query_key)
            if rows is not None:
                self.sql_cache.move_to_end(query_key)
            return rows


class WorldRegistry:
    """Thread-safe per-session world registry with LRU eviction."""

    def __init__(self, *, max_worlds: int = _MAX_WORLDS) -> None:
        self._worlds: OrderedDict[str, SessionWorld] = OrderedDict()
        self._lock = threading.Lock()
        self._max_worlds = max(1, int(max_worlds))

    def get_or_create(self, session_id: str) -> SessionWorld:
        with self._lock:
            world = self._worlds.get(session_id)
            if world is None:
                world = SessionWorld.seeded(session_id)
                self._worlds[session_id] = world
                while len(self._worlds) > self._max_worlds:
                    self._worlds.popitem(last=False)
            else:
                self._worlds.move_to_end(session_id)
            return world

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._worlds.pop(session_id, None)
