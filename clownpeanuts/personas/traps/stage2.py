"""Stage-2 ML classifier — semantic prompt-injection detection.

Resolves the X-017 deferred backlog item. Pairs with the stage-1
regex classifier (classifier.py); the trap layer combines both scores
to decide whether to route an input to the canary template renderer
or pass it through to the LLM backend.

The default model is `protectai/deberta-v3-base-prompt-injection-v2`
exported to ONNX FP32. Operators can pin a different model by
dropping `model.onnx` + tokenizer files into a pack's
`traps/stage2/` directory; absence of that directory disables
stage 2 cleanly (the trap layer falls back to stage-1-only).

Score combination rule (in HeuristicClassifier.classify when stage-2
is loaded):

    combined = max(stage1, stage2)

Max is intentional. Stage-2 catches semantic paraphrases that the
stage-1 regex misses; stage-1 catches deterministic markers (DAN
literals, SQL syntax) that stage-2 can be uncertain about. Either
firing should route through the trap layer. We deliberately do NOT
average — a high stage-2 score with no stage-1 match would be diluted
below threshold.

ReDoS / runaway-input bound: stage-2 truncates input at 1 KiB before
tokenization. DeBERTa-v3 has a 512-token context window; longer
inputs would error or truncate anyway. 1 KiB upper-bounds tokenizer
work in addition to enforcing the model's own limit.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Truncation bound on stage-2 inputs. DeBERTa-v3 has a 512-token
# context window so longer inputs are useless to the model anyway;
# truncating at the character layer makes the tokenizer step cheap.
MAX_STAGE2_CHARS = 1024


@dataclass(frozen=True, slots=True)
class Stage2Verdict:
    """Single-input verdict from the stage-2 ML classifier."""
    score: float  # P(injection) in [0, 1]
    elapsed_ms: float


class Stage2Classifier:
    """ONNX-backed semantic prompt-injection classifier.

    Threadsafe: ONNX Runtime sessions support concurrent inference
    calls. The wrapper holds the session + tokenizer for the lifetime
    of the trap layer; load cost (~1–2 s for the 700 MB FP32 model)
    is paid once at pack-load time.
    """

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = Path(model_dir)
        # Lazy-imported to keep `personas` startup cheap when stage-2
        # is absent. The heavy ORT + transformers import only fires on
        # the first request.
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        logger.info(
            "stage2: loading classifier from %s", str(self._model_dir)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
        self._model = ORTModelForSequenceClassification.from_pretrained(
            self._model_dir
        )
        # Confirm the model's label-id wiring matches what we expect.
        # protectai/deberta-v3-base-prompt-injection-v2 ships
        # {0: "SAFE", 1: "INJECTION"}; if a future model swap reverses
        # the order, we'd silently invert the score. Hard-fail at load
        # time so the operator notices.
        id2label = {int(k): str(v).upper() for k, v in self._model.config.id2label.items()}
        self._injection_index = next(
            (i for i, label in id2label.items() if "INJ" in label),
            None,
        )
        if self._injection_index is None:
            raise ValueError(
                f"stage-2 model at {self._model_dir} has unexpected "
                f"label mapping: {id2label}. Expected one label to "
                f"contain 'INJ' (e.g. 'INJECTION'). Wire a custom "
                f"index in stage2.py if your model differs."
            )
        # ORT sessions are thread-safe per the docs, but the
        # tokenizer's batch_encode_plus is not. Guard tokenization
        # under a lock so we don't corrupt internal state under
        # concurrent inference.
        self._tokenizer_lock = threading.Lock()

    @classmethod
    def from_pack(cls, pack_dir: Path) -> "Stage2Classifier | None":
        """Load stage-2 from a pack's `traps/stage2/` directory.

        Returns None if the directory is absent — stage-2 is
        optional; packs built before X-017 landed run with stage-1
        alone.
        """
        stage2_dir = Path(pack_dir) / "traps" / "stage2"
        if not stage2_dir.is_dir():
            return None
        if not (stage2_dir / "model.onnx").is_file():
            logger.warning(
                "stage2: %s exists but model.onnx is missing — "
                "disabling stage 2", stage2_dir
            )
            return None
        try:
            return cls(stage2_dir)
        except Exception as e:  # noqa: BLE001 — graceful degrade
            logger.exception(
                "stage2: load failed from %s — disabling stage 2 "
                "(stage-1 heuristics still active): %s",
                stage2_dir, e,
            )
            return None

    def score(self, text: str) -> Stage2Verdict:
        """Return P(injection) for a single input string."""
        import time

        import numpy as np

        if len(text) > MAX_STAGE2_CHARS:
            text = text[:MAX_STAGE2_CHARS]

        t0 = time.monotonic()
        with self._tokenizer_lock:
            inputs = self._tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=512,
            )
        outputs = self._model(**inputs)
        logits = outputs.logits[0]
        # Numerically-stable softmax (the logits can be > 30 for
        # confident inputs; raw exp would overflow float32).
        shifted = logits - logits.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        injection_p = float(probs[self._injection_index])
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return Stage2Verdict(score=injection_p, elapsed_ms=elapsed_ms)
