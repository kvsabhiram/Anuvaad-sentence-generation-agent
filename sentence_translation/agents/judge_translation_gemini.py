"""Gemini translation judge: scores each Telugu translation against its English
source, driven by prompts/judge_translation.txt.

The judge's own `decision`/`overall_score` fields are returned for audit but are
NOT the source of truth -- pipeline/validation.py recomputes PASS/FAIL from the
per-dimension scores against config.yaml's judge_thresholds, and additionally
requires the local cross-lingual semantic-similarity floor that the judge has no
say over.

Batch Mode is used for the main pass (roughly half the synchronous price). One
batch job covers many chunks; pipeline/orchestrator.py splits work into
sub-jobs and runs a small pool of them so no single job is oversized.
"""

from __future__ import annotations

import json
import logging
import os
import time

from google import genai
from google.genai import types

from pipeline.retry import call_with_backoff

logger = logging.getLogger("agents.judge_translation")

_client: genai.Client | None = None

TERMINAL_JOB_STATES = (
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment")
        _client = genai.Client(api_key=api_key)
    return _client


def load_system_prompt(prompts_dir: str) -> str:
    path = os.path.join(prompts_dir, "judge_translation.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _judge_payload(records: list[dict]) -> list[dict]:
    return [
        {
            "sentence_id": r["sentence_id"],
            "domain": r["domain"],
            "category": r["category"],
            "word": r["word"],
            "english_sentence": r["english_sentence"],
            "telugu_translation": r["telugu_sentence"],
        }
        for r in records
    ]


def _build_user_message(records: list[dict]) -> str:
    return (
        "Evaluate every record in `records` below according to your instructions.\n"
        'Return a single JSON object of shape {"evaluations": [<one evaluation '
        "object per input record, in the same order as the input records>]}.\n"
        "Do not include anything outside that one JSON object.\n\n"
        f"records = {json.dumps(_judge_payload(records), ensure_ascii=False)}"
    )


def _generation_config(config: dict, system_prompt: str) -> types.GenerateContentConfig:
    judge_cfg = config["judge"]
    kwargs = dict(
        system_instruction=system_prompt,
        temperature=judge_cfg.get("temperature", 0.0),
        max_output_tokens=judge_cfg.get("max_output_tokens", 32768),
        response_mime_type="application/json",
    )
    # Gemini 3.x replaced the integer thinking_budget with a LOW/MEDIUM/HIGH
    # enum, and defaults to MEDIUM. Measured on 384 sentences: LOW emits exactly
    # zero thinking tokens, leaves output length unchanged (176 vs 277 tokens
    # per judgement), and still agrees with MEDIUM on 96.1% of verdicts --
    # disagreeing in the strict direction. Thinking tokens bill at the output
    # rate, so this is ~40% of the judge bill for no measured gain.
    thinking_level = judge_cfg.get("thinking_level")
    if thinking_level:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    return types.GenerateContentConfig(**kwargs)


def _parse_evaluations(raw_text: str, records: list[dict]) -> dict[str, dict]:
    """Returns {sentence_id: evaluation dict}. Sentences the judge dropped are
    absent; the caller leaves those unresolved rather than assuming a pass."""
    try:
        parsed = json.loads(raw_text)
        evaluations = parsed["evaluations"] if isinstance(parsed, dict) and "evaluations" in parsed else parsed
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.error("judge returned invalid/unexpected JSON: %s", raw_text[:500])
        return {}

    if isinstance(evaluations, dict):
        evaluations = [evaluations]
    if not isinstance(evaluations, list):
        logger.error("judge returned unexpected JSON shape: %s", type(evaluations))
        return {}

    out: dict[str, dict] = {}
    for item in evaluations:
        if isinstance(item, dict) and item.get("sentence_id"):
            out[str(item["sentence_id"])] = item

    missing = [r["sentence_id"] for r in records if r["sentence_id"] not in out]
    if missing:
        logger.warning("judge omitted %d record(s), e.g. %s", len(missing), missing[:5])
    return out


def judge_batch(records: list[dict], config: dict, prompts_dir: str) -> dict[str, dict]:
    """Judge a chunk of translated records via a single synchronous call.
    records: [{sentence_id, domain, category, word, english_sentence,
    telugu_sentence}, ...]."""
    if not records:
        return {}

    judge_cfg = config["judge"]
    retry_cfg = config.get("retry", {})
    system_prompt = load_system_prompt(prompts_dir)
    user_message = _build_user_message(records)

    def _call() -> str:
        client = _get_client()
        response = client.models.generate_content(
            model=judge_cfg["model"],
            contents=user_message,
            config=_generation_config(config, system_prompt),
        )
        if not response.text:
            raise RuntimeError("empty response from judge model")
        return response.text

    raw_text = call_with_backoff(
        _call,
        max_attempts=retry_cfg.get("max_api_attempts", 4),
        base_backoff_seconds=retry_cfg.get("base_backoff_seconds", 2.0),
        max_backoff_seconds=retry_cfg.get("max_backoff_seconds", 30.0),
    )
    return _parse_evaluations(raw_text, records)


# --- Gemini Batch Mode (async, ~50% cheaper than the synchronous call above) ---

def submit_judge_batch_job(chunks: list[list[dict]], config: dict, prompts_dir: str) -> str:
    """chunks: a list of record-lists, each shaped like one judge_batch() input.
    Submits them all as a single Batch Mode job and returns the job name
    immediately (does not wait for completion)."""
    client = _get_client()
    judge_cfg = config["judge"]
    system_prompt = load_system_prompt(prompts_dir)

    requests_ = []
    for i, records in enumerate(chunks):
        requests_.append(types.InlinedRequest(
            model=judge_cfg["model"],
            contents=_build_user_message(records),
            metadata={"chunk_id": f"chunk-{i}"},
            config=_generation_config(config, system_prompt),
        ))

    job = client.batches.create(
        model=judge_cfg["model"],
        src=requests_,
        config=types.CreateBatchJobConfig(display_name="translation-judging"),
    )
    logger.info("submitted judge batch job %s with %d chunk(s)", job.name, len(requests_))
    return job.name


def check_batch_status(job_name: str):
    """Single non-blocking status check, for a caller managing several
    concurrent jobs itself."""
    client = _get_client()
    job = client.batches.get(name=job_name)
    logger.info("judge batch %s state=%s stats=%s", job_name, job.state, job.completion_stats)
    return job


def poll_judge_batch_job(job_name: str, poll_interval_seconds: int = 30, timeout_seconds: int = 24 * 3600):
    """Blocks until the job reaches a terminal state. Safe to call again after a
    process restart with the same job_name -- it reconnects to the existing job."""
    t0 = time.time()
    while True:
        job = check_batch_status(job_name)
        if is_terminal(job):
            return job
        if time.time() - t0 > timeout_seconds:
            raise TimeoutError(f"judge batch job {job_name} did not finish within {timeout_seconds}s")
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


def fetch_judge_batch_results(job, chunks: list[list[dict]]) -> dict[str, dict]:
    """Returns {sentence_id: evaluation} merged across every chunk that came
    back. Failed/missing chunks are simply absent."""
    results: dict[str, dict] = {}
    if not (job.dest and job.dest.inlined_responses):
        logger.error("judge batch job %s produced no inlined_responses (dest=%s)", job.name, job.dest)
        return results

    for r in job.dest.inlined_responses:
        chunk_id = (r.metadata or {}).get("chunk_id")
        if chunk_id is None:
            continue
        idx = int(chunk_id.split("-", 1)[1])
        records = chunks[idx]
        if r.error:
            logger.error("judge batch chunk %s errored: %s", chunk_id, r.error)
            continue
        text = r.response.text if r.response else None
        if not text:
            logger.error("judge batch chunk %s had empty response", chunk_id)
            continue
        results.update(_parse_evaluations(text, records))

    return results
