"""Gemini translator, used in two different modes by two different tiers.

Tier 2 (translate_batch, synchronous): escalation for sentences that failed QC.
A dedicated MT endpoint is effectively deterministic, so re-calling it with the
same input returns the same failing translation -- a retry there changes
nothing. Gemini is a genuinely different translator AND can consume the context
a fixed-parameter MT endpoint has no field for (domain, category, target word)
via prompts/translator.txt. The escalation set is a small fraction of
the corpus, so batch-job overhead would cost more wall-clock than it saves.

Tier 1 (submit_translation_batch_job + fetch_translation_batch_results, async):
the whole-corpus pass, on Gemini's Batch API at roughly half the synchronous
price. Same prompt and same parser as Tier 2 -- only the model, the submission
mechanism and the chunk size differ, so the two tiers cannot drift apart in how
they read a translation out of a response.

The two tiers deliberately run DIFFERENT models (config sets flash-lite for
Tier 1 and a stronger model for Tier 2) so that escalation actually upgrades the
translator instead of re-asking the same model the same question at
temperature 0, which returns the same failing answer.
"""

from __future__ import annotations

import json
import logging
import os
import time

from google import genai
from google.genai import types

from pipeline.retry import call_with_backoff

logger = logging.getLogger("agents.translator_gemini")

TERMINAL_JOB_STATES = (
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment")
        _client = genai.Client(api_key=api_key)
    return _client


def load_system_prompt(prompts_dir: str) -> str:
    path = os.path.join(prompts_dir, "translator.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_user_message(records: list[dict]) -> str:
    payload = [
        {
            "sentence_id": r["sentence_id"],
            "domain": r["domain"],
            "category": r["category"],
            "word": r["word"],
            "english_sentence": r["english_sentence"],
        }
        for r in records
    ]
    return (
        "Translate every record in `records` below according to your instructions.\n"
        'Return a single JSON object of shape {"translations": [<one object per '
        "input record, in the same order as the input records>]}.\n"
        "Do not include anything outside that one JSON object.\n\n"
        f"records = {json.dumps(payload, ensure_ascii=False)}"
    )


def _parse_translations(raw_text: str, records: list[dict]) -> dict[str, str]:
    """Returns {sentence_id: telugu_sentence} for whatever parsed cleanly.
    Records the model dropped or marked FAILED are simply absent -- the caller
    treats a missing sentence_id as an unresolved sentence."""
    try:
        parsed = json.loads(raw_text)
        items = parsed["translations"] if isinstance(parsed, dict) and "translations" in parsed else parsed
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.error("translator returned invalid JSON: %s", raw_text[:500])
        return {}

    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        logger.error("translator returned unexpected JSON shape: %s", type(items))
        return {}

    out: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        sid = item.get("sentence_id")
        telugu = item.get("telugu_sentence")
        if sid and telugu and str(telugu).strip():
            out[str(sid)] = str(telugu).strip()
        elif sid:
            logger.info("translator could not translate %s: %s", sid, item.get("reason", ""))
    return out


def _generation_config(tier_cfg: dict, system_prompt: str) -> types.GenerateContentConfig:
    """Shared by the synchronous and batch paths so a change to decoding
    settings cannot apply to one tier and silently miss the other."""
    kwargs = dict(
        system_instruction=system_prompt,
        temperature=tier_cfg.get("temperature", 0.0),
        max_output_tokens=tier_cfg.get("max_output_tokens", 8192),
        response_mime_type="application/json",
    )
    thinking_level = tier_cfg.get("thinking_level")
    if thinking_level:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    return types.GenerateContentConfig(**kwargs)


def translate_batch(records: list[dict], config: dict, prompts_dir: str) -> dict[str, str]:
    """records: [{sentence_id, domain, category, word, english_sentence}, ...].
    Returns {sentence_id: telugu_sentence}."""
    if not records:
        return {}

    tier2_cfg = config["translation"]["tier2"]
    retry_cfg = config.get("retry", {})
    system_prompt = load_system_prompt(prompts_dir)
    user_message = _build_user_message(records)

    def _call() -> str:
        client = _get_client()
        response = client.models.generate_content(
            model=tier2_cfg["model"],
            contents=user_message,
            config=_generation_config(tier2_cfg, system_prompt),
        )
        if not response.text:
            raise RuntimeError("empty response from tier-2 translator")
        return response.text

    raw_text = call_with_backoff(
        _call,
        max_attempts=retry_cfg.get("max_api_attempts", 4),
        base_backoff_seconds=retry_cfg.get("base_backoff_seconds", 2.0),
        max_backoff_seconds=retry_cfg.get("max_backoff_seconds", 30.0),
    )
    return _parse_translations(raw_text, records)


# --- Tier 1: Gemini Batch Mode (async, ~50% cheaper than the call above) -----

def submit_translation_batch_job(chunks: list[list[dict]], config: dict, prompts_dir: str) -> str:
    """chunks: a list of record-lists, each shaped like one translate_batch()
    input. Submits them as a single Batch Mode job and returns the job name
    immediately, without waiting for completion."""
    client = _get_client()
    tier1_cfg = config["translation"]["tier1"]
    system_prompt = load_system_prompt(prompts_dir)

    requests_ = [
        types.InlinedRequest(
            model=tier1_cfg["model"],
            contents=_build_user_message(records),
            metadata={"chunk_id": f"chunk-{i}"},
            config=_generation_config(tier1_cfg, system_prompt),
        )
        for i, records in enumerate(chunks)
    ]

    job = client.batches.create(
        model=tier1_cfg["model"],
        src=requests_,
        config=types.CreateBatchJobConfig(display_name="tier1-translation"),
    )
    logger.info("submitted tier-1 translation batch job %s with %d chunk(s)", job.name, len(requests_))
    return job.name


def check_batch_status(job_name: str):
    """Single non-blocking status check, for a caller managing several
    concurrent jobs itself."""
    client = _get_client()
    job = client.batches.get(name=job_name)
    logger.info("tier-1 batch %s state=%s stats=%s", job_name, job.state, job.completion_stats)
    return job


def poll_translation_batch_job(job_name: str, poll_interval_seconds: int = 30, timeout_seconds: int = 24 * 3600):
    """Blocks until the job reaches a terminal state. Safe to call again after a
    process restart with the same job_name -- it reconnects to the existing job."""
    t0 = time.time()
    while True:
        job = check_batch_status(job_name)
        if is_terminal(job):
            return job
        if time.time() - t0 > timeout_seconds:
            raise TimeoutError(f"tier-1 batch job {job_name} did not finish within {timeout_seconds}s")
        time.sleep(poll_interval_seconds)


def is_terminal(job) -> bool:
    return job.state.name in TERMINAL_JOB_STATES


def is_batch_usable(job) -> bool:
    """SUCCEEDED and PARTIALLY_SUCCEEDED both carry results worth reading.

    Treating PARTIALLY_SUCCEEDED as a total failure throws away every chunk in
    the job that DID succeed -- and then pays to redo them. The per-response
    error handling in the fetch functions below already skips the individual
    chunks that failed, so reading a partial job is safe.
    """
    return job.state.name in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED")


def fetch_translation_batch_results(job, chunks: list[list[dict]]) -> dict[str, str]:
    """Returns {sentence_id: telugu_sentence} merged across every chunk that came
    back. Failed or dropped chunks are simply absent, and the caller treats a
    missing sentence_id as an unresolved sentence rather than an empty one."""
    results: dict[str, str] = {}
    if not (job.dest and job.dest.inlined_responses):
        logger.error("tier-1 batch job %s produced no inlined_responses (dest=%s)", job.name, job.dest)
        return results

    for r in job.dest.inlined_responses:
        chunk_id = (r.metadata or {}).get("chunk_id")
        if chunk_id is None:
            continue
        idx = int(chunk_id.split("-", 1)[1])
        records = chunks[idx]
        if r.error:
            logger.error("tier-1 batch chunk %s errored: %s", chunk_id, r.error)
            continue
        text = r.response.text if r.response else None
        if not text:
            logger.error("tier-1 batch chunk %s had empty response", chunk_id)
            continue
        results.update(_parse_translations(text, records))
    return results
