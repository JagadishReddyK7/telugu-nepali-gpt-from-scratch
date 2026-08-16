"""
Unicode normalization and script-level filtering for Indic text.

WHY THIS MODULE EXISTS
----------------------
Raw web text in Indic scripts is messy in ways that ASCII text is not:

1. The *same* visible character can have multiple Unicode encodings. Devanagari
   "क़" can be a single precomposed codepoint (U+0958) or the base "क" (U+0915)
   followed by a combining nukta (U+093C). To a human they look identical; to a
   tokenizer they are two completely different strings, so the model would waste
   vocabulary slots learning both. NFC forces one canonical spelling.

   Worth knowing: the Devanagari nukta forms sit on Unicode's *composition
   exclusion* list, so NFC canonicalizes them to the DECOMPOSED form (base +
   combining nukta), not the precomposed one. The direction does not matter for
   us -- what matters is that the choice is consistent across the whole corpus.
   It does mean you should not be surprised when NFC lengthens a string.

2. Web pages are littered with invisible characters -- byte-order marks,
   zero-width spaces, soft hyphens -- that survive HTML extraction and silently
   pollute the vocabulary.

3. Pages nominally in one language routinely contain large runs of another. This
   matters enormously for us: Nepali and Hindi share the Devanagari script, so a
   "Nepali" page can be half Hindi and no script-based check would notice.

A NOTE ON ZWJ / ZWNJ
--------------------
Zero-width joiner (U+200D) and non-joiner (U+200C) are invisible, but in Indic
scripts they are NOT junk -- they control whether adjacent consonants render as a
conjunct ligature. Stripping them changes the orthography. We therefore delete
the genuinely meaningless invisibles (BOM, zero-width space, soft hyphen) while
preserving ZWJ/ZWNJ. This is a deliberate choice and is documented in the report.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Unicode block definitions
# --------------------------------------------------------------------------
# Each Indic script occupies a contiguous block of the Unicode codepoint space.
# Checking "is this character in the block" is the cheapest possible language
# signal -- it cannot distinguish Nepali from Hindi (both Devanagari), but it
# instantly catches English, code, emoji, and other-script contamination.
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    # Telugu: main block only. There is no Telugu supplement block.
    "telugu": [(0x0C00, 0x0C7F)],
    # Devanagari: main block plus the "Devanagari Extended" block, which holds
    # rarely-used vedic accents that do occasionally appear in scraped text.
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
}

# Native digit forms. Kept by default (see NormalizerConfig.map_digits_to_ascii)
# because mapping them away destroys information about how the source actually
# wrote numbers, which is exactly the kind of thing we want the LM to learn.
NATIVE_DIGITS: dict[str, str] = {
    "telugu": "౦౧౨౩౪౫౬౭౮౯",
    "devanagari": "०१२३४५६७८९",
}

# Invisible characters that carry no orthographic meaning and should always go.
# Note the conspicuous absence of U+200C and U+200D -- see module docstring.
JUNK_INVISIBLES = (
    "﻿"  # byte-order mark / zero-width no-break space
    "​"  # zero-width space
    "‎"  # left-to-right mark
    "‏"  # right-to-left mark
    "­"  # soft hyphen
    "⁠"  # word joiner
)
_JUNK_RE = re.compile(f"[{JUNK_INVISIBLES}]")

# Collapse runs of whitespace, but preserve paragraph structure: a document's
# paragraph breaks are real signal for a language model learning discourse.
_SPACES_RE = re.compile(r"[ \t  -   　]+")
_NEWLINES_RE = re.compile(r"\n{3,}")

# Control characters (except tab/newline) that occasionally survive extraction.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class NormalizerConfig:
    """Configuration for :class:`TextNormalizer`.

    Attributes:
        script: Key into :data:`SCRIPT_RANGES` -- "telugu" or "devanagari".
        map_digits_to_ascii: If True, rewrite native digits as 0-9. Default False
            so the corpus preserves the source orthography.
        strip_urls: Remove http(s) URLs and bare www. links, which are pure noise
            for a monolingual LM and inflate the Latin-character ratio.
        strip_emails: Remove email addresses (same reasoning, plus mild privacy).
    """

    script: str
    map_digits_to_ascii: bool = False
    strip_urls: bool = True
    strip_emails: bool = True


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


class TextNormalizer:
    """Deterministically normalize raw scraped/downloaded text.

    The transform is idempotent: normalizing already-normalized text is a no-op.
    That property matters because the pipeline may be re-run on partially
    processed shards after a crash, and we must not get drift between shards.
    """

    def __init__(self, config: NormalizerConfig) -> None:
        if config.script not in SCRIPT_RANGES:
            raise ValueError(
                f"Unknown script {config.script!r}; "
                f"expected one of {sorted(SCRIPT_RANGES)}"
            )
        self.config = config
        self._ranges = SCRIPT_RANGES[config.script]
        self._digit_table = None
        if config.map_digits_to_ascii:
            native = NATIVE_DIGITS[config.script]
            self._digit_table = str.maketrans(native, "0123456789")

    def normalize(self, text: str) -> str:
        """Apply the full normalization chain to one document.

        Args:
            text: Raw document text, typically from HTML extraction or a
                downloaded corpus field.

        Returns:
            Normalized text. May be an empty string if the input was entirely
            junk, so callers must check before writing.
        """
        if not text:
            return ""

        # Order matters here. We strip URLs BEFORE Unicode normalization because
        # percent-encoded URLs can contain sequences that NFC would mangle into
        # something that no longer matches the URL regex.
        if self.config.strip_urls:
            text = _URL_RE.sub(" ", text)
        if self.config.strip_emails:
            text = _EMAIL_RE.sub(" ", text)

        # NFC composition -- the single most important step. See module docstring.
        text = unicodedata.normalize("NFC", text)

        text = _CONTROL_RE.sub("", text)
        text = _JUNK_RE.sub("", text)

        if self._digit_table is not None:
            text = text.translate(self._digit_table)

        # Whitespace cleanup, done last so it also tidies gaps left by the
        # substitutions above.
        text = _SPACES_RE.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = _NEWLINES_RE.sub("\n\n", text)

        return text.strip()

    def script_ratio(self, text: str) -> float:
        """Fraction of *letter* characters that belong to the target script.

        Punctuation, digits, and whitespace are excluded from the denominator.
        This is deliberate: a perfectly good Telugu sentence may be 30% spaces
        and commas, and counting those would drag the ratio down for no reason.
        A document with no letters at all returns 0.0.

        Args:
            text: Text to measure, ideally already normalized.

        Returns:
            Ratio in [0.0, 1.0].
        """
        in_script = 0
        total_letters = 0
        for ch in text:
            if not ch.isalpha():
                continue
            total_letters += 1
            cp = ord(ch)
            if any(lo <= cp <= hi for lo, hi in self._ranges):
                in_script += 1
        if total_letters == 0:
            return 0.0
        return in_script / total_letters


def count_chars(text: str) -> int:
    """Return the number of Unicode codepoints in ``text``."""
    return len(text)


def count_bytes(text: str) -> int:
    """Return the UTF-8 byte length of ``text``.

    We track this separately from character count because Phase 2 requires
    bits-per-byte (BPB), which normalizes away tokenizer differences by dividing
    total loss by the raw byte count of the held-out text. Telugu and Devanagari
    both encode at 3 bytes per character in UTF-8, whereas ASCII is 1, so byte
    counts and character counts diverge sharply and must not be conflated.
    """
    return len(text.encode("utf-8"))
