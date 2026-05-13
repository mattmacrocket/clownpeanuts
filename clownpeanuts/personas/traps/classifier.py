"""Two-stage classifier for the trap layer.

Per spec §5.2:
- Stage 1: regex heuristics, additive scoring. Sub-millisecond.
- Stage 2: DeBERTa-v3-base ONNX confirmation (X-017). Optional —
  loaded from `traps/stage2/` in the pack when present, otherwise
  the trap layer routes on stage 1 alone.

The two stages combine via `max(stage1, stage2)` so either firing
above threshold routes through the trap layer. Stage 1 catches
deterministic markers (DAN literals, SQL syntax); stage 2 catches
semantic paraphrases the regex misses.

Output labels: `benign | probing | jailbreak_attempt | exploit_chain`.

Score → label mapping (against the combined score):
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

from .stage2 import Stage2Classifier


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


def _compile_rules(
    raw_rules: list[dict],
    *,
    source: str,
    name_prefix: str,
) -> list[HeuristicRule]:
    """Compile a list of raw rule dicts into `HeuristicRule` objects.

    `source` ("pack" or "operator") is woven into error messages so
    misconfigured operator rules don't get blamed on the pack and
    vice-versa. `name_prefix` (e.g. "tenant:") is prepended to the
    compiled rule names so attribution in `matched_rules` output is
    unambiguous.

    Validation:
    - `name` and `regex` are required
    - regex must compile (re.error → ValueError naming the rule)
    - score must be a float in (0.0, 1.0]; out-of-range is a hard error
    """
    out: list[HeuristicRule] = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            raise ValueError(
                f"{source} classifier rule is not a mapping: {raw!r}"
            )
        name = raw.get("name")
        regex = raw.get("regex")
        if not name or not regex:
            raise ValueError(
                f"{source} classifier rule missing required name/regex: {raw!r}"
            )
        try:
            pattern = re.compile(regex)
        except re.error as e:
            raise ValueError(
                f"{source} classifier rule {name!r}: bad regex: {e}"
            ) from e
        score = float(raw.get("score", 0.0))
        if not (0.0 < score <= 1.0):
            raise ValueError(
                f"{source} classifier rule {name!r}: score must be in "
                f"(0.0, 1.0], got {score}"
            )
        out.append(
            HeuristicRule(
                name=f"{name_prefix}{name}",
                pattern=pattern,
                score=score,
            )
        )
    return out


class HeuristicClassifier:
    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        rules: list[HeuristicRule],
        threshold: float,
        stage2: "Stage2Classifier | None" = None,
    ) -> None:
        self.rules = rules
        self.threshold = threshold
        self.stage2 = stage2

    @classmethod
    def from_pack(
        cls,
        pack_dir: Path,
        *,
        extra_rules: list[dict] | None = None,
    ) -> "HeuristicClassifier":
        """Build a classifier from the pack-shipped rules, optionally
        appending operator-supplied `extra_rules` (X-018).

        Operator rules are *additive*: they layer on top of the pack
        rules and cannot disable them (by design — silencing pack rules
        is high-risk for a vuln_llm deployment). Each operator rule
        gets a `tenant:` name prefix so it's distinguishable from
        pack rules in `matched_rules` output and in CP intel events.

        Raises ValueError with the offending rule's name if any regex
        won't compile or any score is out of (0.0, 1.0]. We hard-fail
        at load time rather than at first-classify so operator config
        problems surface at service-start, not at request-time.
        """
        path = pack_dir / "traps" / "classifiers.yaml"
        threshold: float = cls.DEFAULT_THRESHOLD
        rules: list[HeuristicRule] = []

        if path.is_file():
            try:
                doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"classifiers.yaml parse error: {e}") from e

            h = doc.get("heuristics", {}) or {}
            threshold = float(h.get("threshold", cls.DEFAULT_THRESHOLD))
            rules.extend(
                _compile_rules(
                    h.get("rules", []) or [],
                    source="pack",
                    name_prefix="",
                )
            )

        if extra_rules:
            rules.extend(
                _compile_rules(
                    extra_rules,
                    source="operator",
                    name_prefix="tenant:",
                )
            )

        return cls(
            rules=rules,
            threshold=threshold,
            stage2=Stage2Classifier.from_pack(pack_dir),
        )

    # Hard cap on the input length passed to regex engines. Python's
    # `re` module has no per-match timeout; pathological patterns over
    # very long inputs can hang a worker thread indefinitely (ReDoS).
    # 8 KiB is generous for legitimate prompt-injection attempts and
    # leaves the remaining 248 KiB of the 256 KiB body cap unmatched.
    MAX_INPUT_CHARS = 8 * 1024

    # Stage-2 score threshold for the "ml signal" pseudo-rule entry.
    # Anything above this counts as an ML-flagged injection; the
    # actual score (not just flag) feeds into the combined score.
    STAGE2_MATCH_THRESHOLD = 0.5

    def classify(self, text: str) -> ClassifierVerdict:
        # Truncate before regex matching to bound worst-case backtracking
        # cost across all rules. Truncation mid-input is fine for our
        # heuristics: if a jailbreak doesn't fit in the first 8 KiB the
        # backend will see the full input anyway, and the routing is
        # already conservative.
        if len(text) > self.MAX_INPUT_CHARS:
            text = text[: self.MAX_INPUT_CHARS]

        # Stage 1: regex heuristics with additive scoring.
        stage1_score = 0.0
        matched: list[str] = []
        for rule in self.rules:
            if rule.pattern.search(text):
                stage1_score += rule.score
                matched.append(rule.name)

        # Stage 2: optional ML score. Combined via max so either stage
        # firing above threshold routes through the trap layer.
        combined_score = stage1_score
        if self.stage2 is not None:
            verdict2 = self.stage2.score(text)
            if verdict2.score >= self.STAGE2_MATCH_THRESHOLD:
                # Record the ML hit with its score so operators can
                # tell from the matched_rules whether stage-1 or
                # stage-2 (or both) drove the verdict.
                matched.append(f"stage2_ml({verdict2.score:.2f})")
            combined_score = max(combined_score, verdict2.score)

        if combined_score < 0.3:
            label = "benign"
        elif combined_score < self.threshold:
            label = "probing"
        elif combined_score < 1.0:
            label = "jailbreak_attempt"
        else:
            label = "exploit_chain"

        return ClassifierVerdict(
            label=label,
            score=round(combined_score, 3),
            matched_rules=tuple(matched),
        )
