"""Tabular export of the final English->Telugu dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd

FINAL_COLUMNS = [
    "sentence_id",
    "row_id",
    "domain",
    "category",
    "word",
    "source_language",
    "target_language",
    "english_sentence",
    "telugu_sentence",
    "telugu_romanized",
    "translation_provider",
    "translation_model",
    "judge_provider",
    "judge_model",
    "meaning_preservation",
    "completeness",
    "hallucination",
    "target_word_accuracy",
    "grammar",
    "naturalness",
    "context_preservation",
    "overall_score",
    "semantic_similarity",
    "status",
]


def write_final_excel(path: str, records: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(records, columns=FINAL_COLUMNS)
    df.to_excel(path, index=False)
