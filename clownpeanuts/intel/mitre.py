"""MITRE ATT&CK mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(slots=True)
class TechniqueMatch:
    technique_id: str
    technique_name: str
    tactic: str
    confidence: float


_TECHNIQUE_CATALOG: tuple[dict[str, str], ...] = (
    {"technique_id": "T1110", "technique_name": "Brute Force", "tactic": "Credential Access"},
    {"technique_id": "T1082", "technique_name": "System Information Discovery", "tactic": "Discovery"},
    {"technique_id": "T1046", "technique_name": "Network Service Discovery", "tactic": "Discovery"},
    {"technique_id": "T1105", "technique_name": "Ingress Tool Transfer", "tactic": "Command and Control"},
    {"technique_id": "T1190", "technique_name": "Exploit Public-Facing Application", "tactic": "Initial Access"},
    {"technique_id": "T1059", "technique_name": "Command and Scripting Interpreter", "tactic": "Execution"},
    {"technique_id": "T1021", "technique_name": "Remote Services", "tactic": "Lateral Movement"},
    {"technique_id": "T1005", "technique_name": "Data from Local System", "tactic": "Collection"},
    {"technique_id": "T1041", "technique_name": "Exfiltration Over C2 Channel", "tactic": "Exfiltration"},
    # LLM attack mappings (HDL vuln_llm service). Cross-walk:
    #   classifier_result(jailbreak_attempt | exploit_chain) → T1190
    #     + AML.T0051 (LLM Prompt Injection, MITRE ATLAS)
    #     + AML.T0054 (LLM Jailbreak, MITRE ATLAS) for exploit_chain
    #   tool_called(execute_query)                          → T1059
    #   tool_called(read_file)                              → T1005
    #   tool_called(list_secrets)                           → T1552
    #   tool_called(query_user_db)                          → T1213
    #
    # Note: canary_issued is intentionally NOT mapped to T1606
    # ("Forge Web Credentials"). T1606 describes the ADVERSARY forging
    # credentials. Here the DEFENDER forges canaries to hand to the
    # attacker. The matching adversary action (T1078 Valid Accounts)
    # would only fire when the adversary later USES the canary
    # somewhere else — and we don't observe that in this loop. The
    # downstream canary-hit pipeline is the right place for it.
    {"technique_id": "T1552", "technique_name": "Unsecured Credentials", "tactic": "Credential Access"},
    {"technique_id": "T1213", "technique_name": "Data from Information Repositories", "tactic": "Collection"},
    # MITRE ATLAS — community-maintained ATT&CK extension for ML/AI systems.
    # IDs use the AML.* namespace and are intentionally separate from the
    # T#### catalog so they are still distinguishable in coverage reports.
    {"technique_id": "AML.T0051", "technique_name": "LLM Prompt Injection", "tactic": "Initial Access"},
    {"technique_id": "AML.T0054", "technique_name": "LLM Jailbreak", "tactic": "Defense Evasion"},
)


def map_event_to_techniques(event: dict[str, Any]) -> list[TechniqueMatch]:
    service = str(event.get("service", ""))
    action = str(event.get("action", ""))
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    matches: list[TechniqueMatch] = []
    if action in {"auth_attempt", "credential_capture"}:
        matches.append(
            TechniqueMatch(
                technique_id="T1110",
                technique_name="Brute Force",
                tactic="Credential Access",
                confidence=0.92,
            )
        )

    command_text = _extract_command_text(payload)
    if command_text:
        if re.search(r"\b(whoami|id|uname|hostname)\b", command_text):
            matches.append(
                TechniqueMatch(
                    technique_id="T1082",
                    technique_name="System Information Discovery",
                    tactic="Discovery",
                    confidence=0.82,
                )
            )
        if re.search(r"/etc/passwd|select\s+.+from|listdatabases|show\s+tables", command_text):
            matches.append(
                TechniqueMatch(
                    technique_id="T1046",
                    technique_name="Network Service Discovery",
                    tactic="Discovery",
                    confidence=0.67,
                )
            )
        if re.search(r"\bcurl\b|\bwget\b|\bscp\b|\brsync\b", command_text):
            matches.append(
                TechniqueMatch(
                    technique_id="T1105",
                    technique_name="Ingress Tool Transfer",
                    tactic="Command and Control",
                    confidence=0.74,
                )
            )

    if service.startswith("http") and action in {"http_request", "credential_capture"}:
        matches.append(
            TechniqueMatch(
                technique_id="T1190",
                technique_name="Exploit Public-Facing Application",
                tactic="Initial Access",
                confidence=0.61,
            )
        )

    if service.endswith("db") and action == "command":
        matches.append(
            TechniqueMatch(
                technique_id="T1059",
                technique_name="Command and Scripting Interpreter",
                tactic="Execution",
                confidence=0.45,
            )
        )

    # ---------- LLM attack mappings (HDL vuln_llm service) ----------
    if service == "vuln_llm":
        matches.extend(_map_vuln_llm_event(action=action, payload=payload))

    unique: dict[str, TechniqueMatch] = {}
    for match in matches:
        prev = unique.get(match.technique_id)
        if prev is None or prev.confidence < match.confidence:
            unique[match.technique_id] = match
    return list(unique.values())


def summarize_techniques(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for event in events:
        for match in map_event_to_techniques(event):
            current = summary.get(match.technique_id)
            if current is None:
                summary[match.technique_id] = {
                    "technique_id": match.technique_id,
                    "technique_name": match.technique_name,
                    "tactic": match.tactic,
                    "count": 1,
                    "confidence": match.confidence,
                }
                continue
            current["count"] += 1
            current["confidence"] = max(float(current["confidence"]), match.confidence)
    return sorted(summary.values(), key=lambda item: str(item["technique_id"]))


def summarize_coverage(techniques: list[dict[str, Any]]) -> dict[str, Any]:
    observed_by_id: dict[str, dict[str, Any]] = {}
    for item in techniques:
        if not isinstance(item, dict):
            continue
        technique_id = str(item.get("technique_id", "")).strip()
        if not technique_id:
            continue
        observed_by_id[technique_id] = {
            "technique_id": technique_id,
            "technique_name": str(item.get("technique_name", "")),
            "tactic": str(item.get("tactic", "")),
            "count": int(item.get("count", 0) or 0),
            "confidence": float(item.get("confidence", 0.0) or 0.0),
        }

    catalog_by_id = {item["technique_id"]: item for item in _TECHNIQUE_CATALOG}
    catalog_size = len(_TECHNIQUE_CATALOG)
    observed_catalog_ids = sorted(set(observed_by_id.keys()) & set(catalog_by_id.keys()))
    observed_count = len(observed_catalog_ids)
    coverage_percent = round((observed_count / catalog_size) * 100.0, 2) if catalog_size else 0.0

    observed: list[dict[str, Any]] = []
    for technique_id in observed_catalog_ids:
        merged = dict(catalog_by_id[technique_id])
        merged.update(observed_by_id[technique_id])
        observed.append(merged)
    observed.sort(key=lambda item: str(item["technique_id"]))

    gaps: list[dict[str, Any]] = []
    for item in _TECHNIQUE_CATALOG:
        technique_id = item["technique_id"]
        if technique_id not in observed_by_id:
            gaps.append(dict(item))

    tactic_totals: dict[str, int] = {}
    tactic_observed: dict[str, int] = {}
    for item in _TECHNIQUE_CATALOG:
        tactic = item["tactic"]
        tactic_totals[tactic] = tactic_totals.get(tactic, 0) + 1
    for technique_id in observed_catalog_ids:
        tactic = str(catalog_by_id[technique_id]["tactic"])
        tactic_observed[tactic] = tactic_observed.get(tactic, 0) + 1

    tactics = []
    for tactic, total in sorted(tactic_totals.items()):
        observed_value = tactic_observed.get(tactic, 0)
        tactics.append(
            {
                "tactic": tactic,
                "observed": observed_value,
                "total": total,
                "coverage_percent": round((observed_value / total) * 100.0, 2) if total else 0.0,
            }
        )

    return {
        "catalog_size": catalog_size,
        "observed_count": observed_count,
        "coverage_percent": coverage_percent,
        "observed": observed,
        "gaps": gaps,
        "tactics": tactics,
    }


def _extract_command_text(payload: dict[str, Any]) -> str:
    fragments: list[str] = []
    for key in ("command", "query", "path", "verb"):
        value = payload.get(key)
        if isinstance(value, str):
            fragments.append(value.lower())
        elif isinstance(value, list):
            fragments.extend(str(item).lower() for item in value)
    return " ".join(fragments)


# ---------- vuln_llm → ATT&CK mapping ----------
#
# HDL's vuln_llm emulator emits CP-native events with these actions:
#   - classifier_result: payload.label ∈ {benign, jailbreak_attempt, exploit_chain}
#   - canary_issued:     payload.canary_type ∈ {aws, code, http, dns, email}
#   - tool_called:       payload.tool ∈ {query_user_db, read_file, execute_query,
#                                         list_secrets}
#   - turn_received / turn_responded / backend_error / service_start / service_stop
#
# We map adversary intent (not the emulator's reactive deception) to ATT&CK:
# the user input *attempts* the technique; the canary/tool response is the
# decoy. So a `jailbreak_attempt` verdict means the attacker is trying to
# Exploit a Public-Facing Application (the LLM), and a `tool_called` event
# means the attacker successfully induced a tool invocation (still
# synthetic, but the intent matches the technique).


def _map_vuln_llm_event(
    *, action: str, payload: dict[str, Any]
) -> list[TechniqueMatch]:
    matches: list[TechniqueMatch] = []

    if action == "classifier_result":
        label = str(payload.get("label", ""))
        if label in {"jailbreak_attempt", "exploit_chain"}:
            matches.append(
                TechniqueMatch(
                    technique_id="T1190",
                    technique_name="Exploit Public-Facing Application",
                    tactic="Initial Access",
                    confidence=0.78 if label == "exploit_chain" else 0.62,
                )
            )
            # MITRE ATLAS: prompt injection is the ML-specific framing.
            matches.append(
                TechniqueMatch(
                    technique_id="AML.T0051",
                    technique_name="LLM Prompt Injection",
                    tactic="Initial Access",
                    confidence=0.85 if label == "exploit_chain" else 0.70,
                )
            )
        if label == "exploit_chain":
            # Multi-stage chains imply the attacker is also trying scripted
            # execution against the model's tool surface.
            matches.append(
                TechniqueMatch(
                    technique_id="T1059",
                    technique_name="Command and Scripting Interpreter",
                    tactic="Execution",
                    confidence=0.55,
                )
            )
            # MITRE ATLAS: jailbreak (multi-step pattern: DAN, refusal
            # bypass, role-play coercion, etc.).
            matches.append(
                TechniqueMatch(
                    technique_id="AML.T0054",
                    technique_name="LLM Jailbreak",
                    tactic="Defense Evasion",
                    confidence=0.80,
                )
            )

    elif action == "tool_called":
        tool = str(payload.get("tool", ""))
        if tool == "execute_query":
            matches.append(
                TechniqueMatch(
                    technique_id="T1059",
                    technique_name="Command and Scripting Interpreter",
                    tactic="Execution",
                    confidence=0.71,
                )
            )
        elif tool == "read_file":
            matches.append(
                TechniqueMatch(
                    technique_id="T1005",
                    technique_name="Data from Local System",
                    tactic="Collection",
                    confidence=0.68,
                )
            )
        elif tool == "list_secrets":
            matches.append(
                TechniqueMatch(
                    technique_id="T1552",
                    technique_name="Unsecured Credentials",
                    tactic="Credential Access",
                    confidence=0.84,
                )
            )
        elif tool == "query_user_db":
            matches.append(
                TechniqueMatch(
                    technique_id="T1213",
                    technique_name="Data from Information Repositories",
                    tactic="Collection",
                    confidence=0.66,
                )
            )

    # canary_issued is a DEFENDER action (we forge a token to hand to the
    # attacker). It does not map to any adversary technique here. The
    # corresponding adversary technique (T1078 Valid Accounts, when the
    # canary is later used somewhere else) is mapped by the downstream
    # canary-hit pipeline, not here.

    return matches
