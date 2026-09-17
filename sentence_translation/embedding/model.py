"""Local cross-lingual embedding model, used for two things: scoring how
semantically close a Telugu translation is to its English source, and
near-duplicate detection among a word's Telugu translations.

Runs on-GPU via sentence-transformers rather than a paid embedding API -- it is
called for every one of ~158k sentence pairs, so keeping it local and free
matters at this scale.

The model must be multilingual: English and Telugu have to land in the same
vector space for their cosine similarity to mean anything. e5 models are
trained with "query: " / "passage: " prefixes and lose accuracy without them,
so callers pass the prefixes from config.
"""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache[model_name] = SentenceTransformer(model_name, device=device)
    return _model_cache[model_name]


def embed_texts(
    texts: list[str],
    model: str = "intfloat/multilingual-e5-base",
    prefix: str = "",
) -> list[list[float]]:
    """Embed a batch of strings. Returns one vector per input, same order.

    Vectors are L2-normalized so cosine similarity is a plain dot product and
    similarity values stay comparable across calls.
    """
    if not texts:
        return []
    m = _get_model(model)
    prefixed = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
    vectors = m.encode(
        prefixed,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def embed_text(
    text: str,
    model: str = "intfloat/multilingual-e5-base",
    prefix: str = "",
) -> list[float]:
    return embed_texts([text], model=model, prefix=prefix)[0]
