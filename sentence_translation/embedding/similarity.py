"""Cosine-similarity helpers over the local cross-lingual embeddings.

Two distinct jobs live here:
  1. semantic_similarity_scores() -- how close each Telugu translation is to
     its own English source. This is an independent signal from the LLM judge:
     the judge could hallucinate a high score for a translation that drifted,
     but the embedding comparison cannot be talked into agreeing.
  2. find_near_duplicate() -- whether two Telugu translations under the SAME
     word are near-identical. This module is the sole authority on that; the
     judge's opinion about duplicates is not consulted.
"""

from __future__ import annotations

import numpy as np

from embedding.model import embed_texts


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_similarity_scores(
    pairs: list[tuple[str, str]],
    model: str,
    query_prefix: str = "",
    passage_prefix: str = "",
) -> list[float]:
    """Score a batch of (english_source, telugu_translation) pairs.

    Both sides are embedded in one batched call each rather than pair by pair,
    which matters at 158k pairs.
    """
    if not pairs:
        return []
    english = [p[0] for p in pairs]
    telugu = [p[1] for p in pairs]
    english_vecs = embed_texts(english, model=model, prefix=query_prefix)
    telugu_vecs = embed_texts(telugu, model=model, prefix=passage_prefix)
    return [cosine_similarity(e, t) for e, t in zip(english_vecs, telugu_vecs)]


def find_near_duplicate(
    candidate_embedding: list[float],
    existing_embeddings: list[list[float]],
    threshold: float,
) -> int | None:
    """Index of the first existing embedding that is a near-duplicate of the
    candidate, or None if the candidate is sufficiently distinct."""
    for idx, existing in enumerate(existing_embeddings):
        if cosine_similarity(candidate_embedding, existing) >= threshold:
            return idx
    return None


def embed_translations(
    translations: list[str],
    model: str,
    passage_prefix: str = "",
) -> list[list[float]]:
    """Embed Telugu translations for duplicate checking, using the same prefix
    convention as the passage side of the semantic comparison."""
    return embed_texts(translations, model=model, prefix=passage_prefix)
