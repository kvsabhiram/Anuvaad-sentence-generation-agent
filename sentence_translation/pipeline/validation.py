"""Deterministic validation that sits between the translation agents and the
pipeline state/storage layers.

Two things are intentionally NOT trusted from the models:
  1. The translator's output is rule-checked locally (script, leakage, length,
     entity preservation) BEFORE it is allowed to cost a judge call. A model
     asked politely to preserve numbers will still occasionally drop one, so
     these rules verify the translation prompt's requirements rather than
     trusting that prompting them was enough.
  2. The judge's self-reported `decision`/`overall_score` are recorded for
     audit only; the real PASS/FAIL gate is recomputed here from the judge's
     per-dimension scores against config.yaml's judge_thresholds.

Near-duplicate rejection is NOT done here -- that lives in
embedding/similarity.py, which pipeline/orchestrator.py calls directly.
"""

from __future__ import annotations

import re

DIMENSION_KEYS = [
    "meaning_preservation",
    "completeness",
    "hallucination",
    "target_word_accuracy",
    "grammar",
    "naturalness",
    "context_preservation",
]

# Telugu block (U+0C00-U+0C7F). Vowel signs/virama live in here too, so a
# well-formed Telugu sentence is almost entirely inside this range.
_TELUGU_RE = re.compile(r"[ఀ-౿]")
_LATIN_RUN_RE = re.compile(r"[A-Za-z]+")
_DIGIT_RUN_RE = re.compile(r"\d+")


def telugu_letter_ratio(text: str) -> float:
    """Share of alphabetic characters that are Telugu. Digits, punctuation and
    whitespace are excluded from both sides so that "15%" or a preserved brand
    name doesn't drag the ratio down on its own."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    telugu = sum(1 for ch in letters if _TELUGU_RE.match(ch))
    return telugu / len(letters)


def longest_latin_run(text: str) -> int:
    """Longest unbroken run of Latin letters, ignoring spaces between words.

    A preserved proper noun ("Bluetooth") is short; an untranslated English
    clause is long. Runs are measured per whitespace-separated span so that a
    multi-word English phrase counts as one long run rather than several short
    ones."""
    longest = 0
    for span in re.split(r"[^A-Za-z\s]+", text):
        letters_only = re.sub(r"\s+", "", span)
        if _LATIN_RUN_RE.fullmatch(letters_only or "x") and letters_only:
            longest = max(longest, len(letters_only))
    return longest


def digits_preserved(english: str, telugu: str) -> tuple[bool, list[str]]:
    """Every digit sequence in the source must survive into the translation.

    Only checks presence, not position -- Telugu word order differs from
    English, so a positional check would produce false failures."""
    source_digits = _DIGIT_RUN_RE.findall(english)
    if not source_digits:
        return True, []
    missing = [d for d in source_digits if d not in telugu]
    return (not missing), missing


def check_translation_rules(
    english: str,
    telugu: str,
    rules: dict,
) -> tuple[bool, str]:
    """Deterministic gate applied to every translation before it reaches the
    judge. Returns (passed, reason)."""
    if not telugu or not telugu.strip():
        return False, "empty translation"

    telugu = telugu.strip()

    ratio = telugu_letter_ratio(telugu)
    min_ratio = rules["min_telugu_letter_ratio"]
    if ratio < min_ratio:
        return False, f"Telugu letter ratio {ratio:.2f} < {min_ratio:.2f} (not Telugu script / romanized)"

    latin_run = longest_latin_run(telugu)
    max_run = rules["max_latin_run_chars"]
    if latin_run > max_run:
        return False, f"untranslated English leakage: {latin_run}-char Latin run > {max_run}"

    english_len = len(english.strip())
    if english_len:
        length_ratio = len(telugu) / english_len
        if length_ratio < rules["min_length_ratio"]:
            return False, f"translation too short: length ratio {length_ratio:.2f} < {rules['min_length_ratio']:.2f}"
        if length_ratio > rules["max_length_ratio"]:
            return False, f"translation too long (possible repetition): length ratio {length_ratio:.2f} > {rules['max_length_ratio']:.2f}"

    if rules.get("require_digit_preservation", True):
        ok, missing = digits_preserved(english, telugu)
        if not ok:
            return False, f"numbers dropped from translation: {missing}"

    return True, ""


def compute_translation_decision(
    scores: dict,
    thresholds: dict,
    semantic_similarity: float | None,
    semantic_threshold: float,
) -> tuple[str, float, str]:
    """Recompute PASS/FAIL deterministically from the judge's per-dimension
    scores plus the local cross-lingual semantic similarity.

    Returns (decision, overall_score, reason). overall_score is the plain mean
    of the seven judge dimensions, computed here rather than trusted from the
    judge's own response.
    """
    missing = [k for k in DIMENSION_KEYS if k not in scores]
    if missing:
        return "FAIL", 0.0, f"judge response missing dimension(s): {missing}"

    values = [float(scores[k]) for k in DIMENSION_KEYS]
    overall_score = sum(values) / len(values)

    failures = [
        f"{dim} {float(scores[dim]):.2f} < {thresholds[dim]:.2f}"
        for dim in DIMENSION_KEYS
        if dim in thresholds and float(scores[dim]) < thresholds[dim]
    ]

    if semantic_similarity is None:
        failures.append("semantic similarity unavailable")
    elif semantic_similarity < semantic_threshold:
        failures.append(f"semantic_similarity {semantic_similarity:.2f} < {semantic_threshold:.2f}")

    if failures:
        return "FAIL", overall_score, "; ".join(failures)
    return "PASS", overall_score, ""
