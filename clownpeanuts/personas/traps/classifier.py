"""Stage-1 (regex-only) classifier for the trap layer.

Per spec §5.2, classification is two-stage:
- Stage 1: regex heuristics, additive scoring. Sub-millisecond.
- Stage 2: DeBERTa-v3-base ONNX confirmation. **DEFERRED** to a follow-up
  milestone — for M2 we route on stage 1 alone. The threshold from
  classifiers.yaml is treated as the decision boundary.

Output labels: `benign | probing | jailbreak_attempt | exploit_chain`.

Score → label mapping (heuristic-only, will be overridden by stage 2 once
wired):
- score < 0.3            → benign
- 0.3 ≤ score < threshold → probing
- threshold ≤ score < 1.0 → jailbreak_attempt
- score ≥ 1.0             → exploit_chain
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class HeuristicRule:
    name: str
    pattern: re.Pattern[str]
    score: float


@dataclass(frozen=True, slots=True)
class ClassifierVerdict:
    label: str  # benign | probing | jailbreak_attempt | exploit_chain
    score: float
    matched_rules: tuple[str, ...]


class HeuristicClassifier:
    DEFAULT_THRESHOLD = 0.5

    def __init__(self, rules: list[HeuristicRule], threshold: float) -> None:
        self.rules = rules
        self.threshold = threshold

    @classmethod
    def from_pack(cls, pack_dir: Path) -> "HeuristicClassifier":
        path = pack_dir / "traps" / "classifiers.yaml"
        if not path.is_file():
            return cls(rules=[], threshold=cls.DEFAULT_THRESHOLD)

        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"classifiers.yaml parse error: {e}") from e

        h = doc.get("heuristics", {}) or {}
        threshold = float(h.get("threshold", cls.DEFAULT_THRESHOLD))

        rules: list[HeuristicRule] = []
        for raw in h.get("rules", []) or []:
            try:
                pattern = re.compile(raw["regex"])
            except re.error as e:
                raise ValueError(
                    f"classifiers.yaml: bad regex in rule {raw.get('name', '?')}: {e}"
                ) from e
            rules.append(
                HeuristicRule(
                    name=str(raw["name"]),
                    pattern=pattern,
                    score=float(raw.get("score", 0.0)),
                )
            )
        return cls(rules=rules, threshold=threshold)

    # Hard cap on the input length passed to regex engines. Python's
    # `re` module has no per-match timeout; pathological patterns over
    # very long inputs can hang a worker thread indefinitely (ReDoS).
    # 8 KiB is generous for legitimate prompt-injection attempts and
    # leaves the remaining 248 KiB of the 256 KiB body cap unmatched.
    MAX_INPUT_CHARS = 8 * 1024

    def classify(self, text: str) -> ClassifierVerdict:
        # Truncate before regex matching to bound worst-case backtracking
        # cost across all rules. Truncation mid-input is fine for our
        # heuristics: if a jailbreak doesn't fit in the first 8 KiB the
        # backend will see the full input anyway, and the routing is
        # already conservative.
        if len(text) > self.MAX_INPUT_CHARS:
            text = text[: self.MAX_INPUT_CHARS]

        score = 0.0
        matched: list[str] = []
        for rule in self.rules:
            if rule.pattern.search(text):
                score += rule.score
                matched.append(rule.name)

        if score < 0.3:
            label = "benign"
        elif score < self.threshold:
            label = "probing"
        elif score < 1.0:
            label = "jailbreak_attempt"
        else:
            label = "exploit_chain"

        return ClassifierVerdict(
            label=label, score=round(score, 3), matched_rules=tuple(matched)
        )
