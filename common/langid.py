"""
Document-level language identification.

WHY THIS IS THE MOST IMPORTANT FILTER IN THE NEPALI PIPELINE
------------------------------------------------------------
Nepali and Hindi are both written in Devanagari. They share a script, a large
amount of Sanskrit-derived vocabulary, and much of their punctuation. Any
script-range check (see ``text_norm.script_ratio``) will happily pass a Hindi
document as Nepali, because at the codepoint level it *is* indistinguishable.

This matters beyond data hygiene. The project spec imposes a hard constraint:

    "Build two fully independent monolingual LMs ... Do not train a multilingual
     model or share artifacts across languages."

If our Nepali corpus is 15% Hindi, Model L is a bilingual Hindi-Nepali model
wearing a Nepali label. That is a constraint violation, and it would also
contaminate the Phase 3 analysis -- we would be attributing to "low resource"
what is actually an artifact of training on two languages at once.

Telugu has a unique script, so for Model H this filter is a cheap safety net
rather than a necessity. We run it on both for symmetry and so the report can
quote rejection rates for each.

MODEL CHOICE
------------
fastText's ``lid.176`` is the standard tool: 176 languages, single ~131MB binary
(or ~917KB for the compressed variant), CPU-only, thousands of documents per
second. Crucially it distinguishes ``ne`` (Nepali) from ``hi`` (Hindi) rather
than lumping them as "Devanagari".

We use the FULL model (``lid.176.bin``), not the compressed one. The compressed
variant is quantized and measurably worse on exactly the close-pair distinctions
we care about, which is the whole reason we are here.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
DEFAULT_MODEL_PATH = Path(os.environ.get("LID_MODEL_PATH", "models/lid.176.bin"))


def ensure_lid_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the fastText language-ID model if it is not already present.

    Args:
        path: Where the model binary should live.

    Returns:
        The path to the downloaded model.
    """
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading fastText lid.176 model to %s (~131MB)", path)
    # Download to a temp name then rename, so an interrupted download can never
    # leave a truncated file that later looks valid.
    tmp = path.with_suffix(".partial")
    urllib.request.urlretrieve(LID_MODEL_URL, tmp)
    tmp.rename(path)
    return path


@dataclass
class LangIDResult:
    """Outcome of a language-identification call.

    Attributes:
        label: ISO 639-1 code of the top prediction, e.g. "ne", "hi", "te".
        confidence: Model probability for that label, in [0.0, 1.0].
        accepted: Whether this document passes the configured filter.
        reason: Human-readable explanation, used for the rejection-reason
            histogram in the Phase 1 report.
    """

    label: str
    confidence: float
    accepted: bool
    reason: str


class LanguageFilter:
    """Accept or reject documents by predicted language.

    Args:
        target_lang: ISO code the corpus is supposed to be, e.g. "ne" or "te".
        min_confidence: Minimum probability required to accept. Tuned per
            language -- see the per-language YAML configs. For Nepali we set
            this high (0.70) precisely because Hindi is a plausible competing
            hypothesis and we would rather discard borderline Nepali than admit
            confident-looking Hindi.
        model_path: Location of the fastText binary.
    """

    def __init__(
        self,
        target_lang: str,
        min_confidence: float = 0.70,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        try:
            import fasttext
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "fasttext is required for language filtering. "
                "Install with: pip install fasttext-wheel"
            ) from exc

        self.target_lang = target_lang
        self.min_confidence = min_confidence
        model_file = ensure_lid_model(model_path)
        # fastText prints a spurious warning to stderr on load; harmless.
        self._model = fasttext.load_model(str(model_file))

    def identify(self, text: str) -> LangIDResult:
        """Classify a single document.

        Args:
            text: Document text. Should already be normalized.

        Returns:
            A :class:`LangIDResult`.

        Note:
            fastText treats a newline as a document separator, so we replace
            newlines with spaces before predicting. Forgetting this is a classic
            bug: the model silently classifies only the first line of a long
            article, which for news pages is often just a dateline.
        """
        flat = text.replace("\n", " ").strip()
        if not flat:
            return LangIDResult("unknown", 0.0, False, "empty_after_flatten")

        # Very short strings give unreliable predictions. 40 characters is
        # roughly a short sentence in either script.
        if len(flat) < 40:
            return LangIDResult("unknown", 0.0, False, "too_short_for_langid")

        labels, probs = self._model.predict(flat, k=1)
        label = labels[0].replace("__label__", "")
        confidence = float(probs[0])

        if label != self.target_lang:
            return LangIDResult(
                label, confidence, False, f"wrong_language:{label}"
            )
        if confidence < self.min_confidence:
            return LangIDResult(
                label, confidence, False, "low_confidence"
            )
        return LangIDResult(label, confidence, True, "accepted")

    def identify_batch(self, texts: list[str]) -> list[LangIDResult]:
        """Classify many documents. Convenience wrapper over :meth:`identify`.

        Args:
            texts: Document texts.

        Returns:
            One :class:`LangIDResult` per input, in the same order.
        """
        return [self.identify(t) for t in texts]
