"""ATT&CK mapping for HDL vuln_llm events (M4-001).

Verifies that the mitre.map_event_to_techniques function correctly
classifies the CP-native finding shapes emitted by the vuln_llm
emulator into ATT&CK technique matches.

Spec: hueydeweylouie/docs/HUEYDEWEYLOUIE-SPEC.md §11.
"""

from __future__ import annotations

from clownpeanuts.intel.fingerprints import (
    fingerprint_events,
    summarize_fingerprints,
)
from clownpeanuts.intel.mitre import (
    map_event_to_techniques,
    summarize_coverage,
    summarize_techniques,
)


def _ids(matches) -> set[str]:
    return {m.technique_id for m in matches}


# ---------- classifier_result ----------


def test_classifier_benign_yields_no_techniques() -> None:
    matches = map_event_to_techniques(
        {
            "service": "vuln_llm",
            "action": "classifier_result",
            "payload": {"label": "benign", "score": 0.0, "matched_rules": []},
        }
    )
    assert matches == []


def test_classifier_jailbreak_attempt_maps_to_t1190_and_atlas() -> None:
    matches = map_event_to_techniques(
        {
            "service": "vuln_llm",
            "action": "classifier_result",
            "payload": {
                "label": "jailbreak_attempt",
                "score": 0.6,
                "matched_rules": ["system_prompt_extraction"],
            },
        }
    )
    ids = _ids(matches)
    assert "T1190" in ids
    # MITRE ATLAS prompt-injection technique fires for any jailbreak verdict.
    assert "AML.T0051" in ids
    # Should NOT include T1059 / AML.T0054 for plain jailbreak_attempt;
    # those are reserved for full exploit chains.
    assert "T1059" not in ids
    assert "AML.T0054" not in ids


def test_classifier_exploit_chain_maps_to_full_set() -> None:
    matches = map_event_to_techniques(
        {
            "service": "vuln_llm",
            "action": "classifier_result",
            "payload": {
                "label": "exploit_chain",
                "score": 1.5,
                "matched_rules": [
                    "dan_pattern",
                    "system_prompt_extraction",
                    "tool_exploit_marker",
                ],
            },
        }
    )
    ids = _ids(matches)
    # Standard ATT&CK
    assert "T1190" in ids
    assert "T1059" in ids
    # MITRE ATLAS
    assert "AML.T0051" in ids
    assert "AML.T0054" in ids
    # Higher confidence on T1190 for exploit_chain than for jailbreak_attempt
    t1190 = next(m for m in matches if m.technique_id == "T1190")
    assert t1190.confidence >= 0.7


# ---------- tool_called ----------


def test_tool_execute_query_maps_to_t1059() -> None:
    matches = map_event_to_techniques(
        {
            "service": "vuln_llm",
            "action": "tool_called",
            "payload": {"tool": "execute_query", "verdict": "exploit_chain"},
        }
    )
    assert "T1059" in _ids(matches)


def test_tool_read_file_maps_to_t1005() -> None:
    matches = map_event_to_techniques(
        {
            "service": "vuln_llm",
            "action": "tool_called",
            "payload": {"tool": "read_file"},
        }
    )
    assert "T1005" in _ids(matches)


def test_tool_list_secrets_maps_to_t1552() -> None:
    matches = map_event_to_techniques(
        {
            "service": "vuln_llm",
            "action": "tool_called",
            "payload": {"tool": "list_secrets"},
        }
    )
    assert "T1552" in _ids(matches)
    # Highest confidence — listing secrets is unambiguous credential access
    t1552 = next(m for m in matches if m.technique_id == "T1552")
    assert t1552.confidence >= 0.8


def test_tool_query_user_db_maps_to_t1213() -> None:
    matches = map_event_to_techniques(
        {
            "service": "vuln_llm",
            "action": "tool_called",
            "payload": {"tool": "query_user_db"},
        }
    )
    assert "T1213" in _ids(matches)


def test_tool_unknown_yields_no_match() -> None:
    matches = map_event_to_techniques(
        {
            "service": "vuln_llm",
            "action": "tool_called",
            "payload": {"tool": "made_up_tool"},
        }
    )
    assert matches == []


# ---------- canary_issued is a DEFENDER action — no mapping ----------


def test_canary_issued_does_not_map_to_adversary_techniques() -> None:
    """canary_issued is the DEFENDER forging tokens to hand to the
    attacker. It must NOT appear in the ATT&CK adversary technique list
    because the adversary hasn't done anything credential-access-y yet.
    The corresponding adversary technique fires only when the canary is
    later observed in use (T1078 Valid Accounts via the canary-hit
    pipeline)."""
    matches = map_event_to_techniques(
        {
            "service": "vuln_llm",
            "action": "canary_issued",
            "payload": {
                "template_id": "api_key_aws_style",
                "canary_type": "aws",
                "token_id": "tok-abc",
            },
        }
    )
    assert matches == []


# ---------- non-vuln_llm events should be unaffected ----------


def test_other_services_unaffected() -> None:
    """Existing technique mappings for ssh/http etc. must still work."""
    matches = map_event_to_techniques(
        {
            "service": "ssh",
            "action": "auth_attempt",
            "payload": {"username": "root", "password": "toor"},
        }
    )
    assert "T1110" in _ids(matches)
    # vuln_llm-only mappings should NOT fire on non-vuln_llm events
    assert "T1606" not in _ids(matches)


# ---------- end-to-end: summarize ----------


def test_summarize_techniques_aggregates_vuln_llm_session() -> None:
    """A typical attacker session: probe → jailbreak → tool → canary."""
    events = [
        {
            "service": "vuln_llm",
            "action": "classifier_result",
            "payload": {"label": "benign", "score": 0.0},
        },
        {
            "service": "vuln_llm",
            "action": "classifier_result",
            "payload": {"label": "jailbreak_attempt", "score": 0.7},
        },
        {
            "service": "vuln_llm",
            "action": "classifier_result",
            "payload": {"label": "exploit_chain", "score": 1.6},
        },
        {
            "service": "vuln_llm",
            "action": "tool_called",
            "payload": {"tool": "list_secrets"},
        },
        {
            "service": "vuln_llm",
            "action": "canary_issued",
            "payload": {"template_id": "api_key_aws_style", "canary_type": "aws"},
        },
        {
            "service": "vuln_llm",
            "action": "canary_issued",
            "payload": {"template_id": "internal_url", "canary_type": "http"},
        },
    ]
    summary = summarize_techniques(events)
    ids = {item["technique_id"] for item in summary}
    # Every LLM-attack technique we mapped should appear in the summary
    assert {"T1190", "T1059", "T1552", "AML.T0051", "AML.T0054"} <= ids
    # canary_issued must NOT appear (defender action, see test above)
    assert "T1606" not in ids
    # AML.T0051 fires on every jailbreak/exploit verdict (2 events)
    atlas = next(item for item in summary if item["technique_id"] == "AML.T0051")
    assert atlas["count"] == 2


def test_summarize_coverage_includes_new_techniques() -> None:
    """The new T1552/T1213/AML.T0051/AML.T0054 catalog entries must be
    counted in coverage."""
    coverage = summarize_coverage(
        [
            {"technique_id": "T1552", "technique_name": "Unsecured Credentials",
             "tactic": "Credential Access", "count": 1, "confidence": 0.9},
            {"technique_id": "AML.T0051", "technique_name": "LLM Prompt Injection",
             "tactic": "Initial Access", "count": 1, "confidence": 0.7},
        ]
    )
    ids = {item["technique_id"] for item in coverage["observed"]}
    assert "T1552" in ids
    assert "AML.T0051" in ids
    # 9 ATT&CK + T1552 + T1213 + AML.T0051 + AML.T0054 = 13
    assert coverage["catalog_size"] == 13


# ---------- LLM-attack tool fingerprints ----------


def test_fingerprint_dan_jailbreak() -> None:
    fps = fingerprint_events(
        [
            {
                "service": "vuln_llm",
                "action": "classifier_result",
                "payload": {
                    "label": "jailbreak_attempt",
                    "matched_rules": ["dan_pattern"],
                    "score": 0.6,
                },
            }
        ]
    )
    tools = {item["tool"] for item in fps}
    assert "llm-jailbreak-dan" in tools


def test_fingerprint_prompt_extraction() -> None:
    fps = fingerprint_events(
        [
            {
                "service": "vuln_llm",
                "action": "classifier_result",
                "payload": {
                    "label": "jailbreak_attempt",
                    "matched_rules": ["system_prompt_extraction"],
                    "score": 0.7,
                },
            }
        ]
    )
    tools = {item["tool"] for item in fps}
    assert "llm-prompt-extraction" in tools


def test_fingerprint_role_coercion() -> None:
    fps = fingerprint_events(
        [
            {
                "service": "vuln_llm",
                "action": "classifier_result",
                "payload": {"matched_rules": ["roleplay_injection"]},
            }
        ]
    )
    assert "llm-role-coercion" in {item["tool"] for item in fps}


def test_fingerprint_encoding_evasion() -> None:
    fps = fingerprint_events(
        [
            {
                "service": "vuln_llm",
                "action": "classifier_result",
                "payload": {"matched_rules": ["encoding_evasion"]},
            }
        ]
    )
    assert "llm-encoding-evasion" in {item["tool"] for item in fps}


def test_fingerprint_tool_exploit() -> None:
    fps = fingerprint_events(
        [
            {
                "service": "vuln_llm",
                "action": "classifier_result",
                "payload": {"matched_rules": ["tool_exploit_marker"]},
            }
        ]
    )
    assert "llm-tool-exploit" in {item["tool"] for item in fps}


def test_fingerprint_multi_rule_session() -> None:
    """A multi-stage attacker that fires DAN + system_prompt_extraction +
    tool_exploit across turns should produce three distinct fingerprints."""
    events = [
        {
            "service": "vuln_llm",
            "action": "classifier_result",
            "payload": {"matched_rules": ["dan_pattern"]},
        },
        {
            "service": "vuln_llm",
            "action": "classifier_result",
            "payload": {"matched_rules": ["system_prompt_extraction"]},
        },
        {
            "service": "vuln_llm",
            "action": "classifier_result",
            "payload": {
                "matched_rules": ["tool_exploit_marker", "system_prompt_extraction"],
            },
        },
    ]
    fps = fingerprint_events(events)
    tools = {item["tool"] for item in fps}
    assert {"llm-jailbreak-dan", "llm-prompt-extraction", "llm-tool-exploit"} <= tools


def test_summarize_fingerprints_aggregates_llm_attacks() -> None:
    """Cross-session aggregation: the same attacker tool seen in multiple
    sessions counts up correctly."""
    session_a = fingerprint_events(
        [
            {
                "service": "vuln_llm",
                "action": "classifier_result",
                "payload": {"matched_rules": ["dan_pattern"]},
            }
        ]
    )
    session_b = fingerprint_events(
        [
            {
                "service": "vuln_llm",
                "action": "classifier_result",
                "payload": {"matched_rules": ["dan_pattern"]},
            }
        ]
    )
    summary = summarize_fingerprints([session_a, session_b])
    dan = next(item for item in summary if item["tool"] == "llm-jailbreak-dan")
    assert dan["sessions"] == 2


def test_fingerprint_benign_yields_no_llm_signatures() -> None:
    """A benign session must not trigger LLM-attack fingerprints."""
    fps = fingerprint_events(
        [
            {
                "service": "vuln_llm",
                "action": "classifier_result",
                "payload": {"label": "benign", "matched_rules": []},
            },
            {
                "service": "vuln_llm",
                "action": "turn_received",
                "payload": {"messages_count": 1, "model": "x"},
            },
        ]
    )
    llm_tools = {
        item["tool"]
        for item in fps
        if item["tool"].startswith("llm-")
    }
    assert llm_tools == set()
