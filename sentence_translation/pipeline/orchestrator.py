"""Drives the end-to-end pipeline: read English sentences -> translate (Tier 1)
-> deterministic rule checks -> judge (Gemini) -> cross-lingual semantic check
-> final QC -> export.

Two design points carried over from the sentence-generation pipeline that
preceded this one, both of which exist because they were learned the hard way:

  1. Nothing an LLM says about its own output is trusted. The judge's
     `decision` is recorded but the accept/reject gate is recomputed locally in
     pipeline/validation.py, and it additionally requires a cross-lingual
     embedding similarity floor that the judge has no influence over.

  2. State is checkpointed after every phase so a multi-hour run survives being
     interrupted. Only sentences that are still pending/translated get
     reprocessed on a re-run.

Every model call is Gemini. See config/config.yaml for which model fills which
role -- translation, transliteration and judging are deliberately not all the
same model.

Failure handling is tiered rather than looped, and the tiers must not share a
model. Re-asking one model the same question at temperature 0 returns the same
failing answer, so a sentence that fails QC escalates to a DIFFERENT, stronger
model (Tier 2) that also sees the domain/category/word context, instead of
retrying Tier 1 forever.
"""

from __future__ import annotations

import logging
import os
import time

import yaml
from dotenv import load_dotenv

import agents.judge_translation_gemini as judge_agent
import agents.translator_gemini as gemini_translator
import embedding.similarity as similarity
import pipeline.batching as batching
import pipeline.retry as retry
import pipeline.validation as validation
import storage.excel as excel_storage
import storage.jsonl as jsonl_storage
from pipeline.state import PipelineState

logger = logging.getLogger("pipeline.orchestrator")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    logs_dir = config["paths"]["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(logs_dir, log_cfg.get("file", "pipeline.log")), encoding="utf-8"),
        ],
        force=True,
    )


def load_input_sentences(path: str) -> list[dict]:
    """Reads the English sentence corpus produced by the generation pipeline and
    normalizes it into this pipeline's record shape."""
    records = []
    for row in jsonl_storage.iter_jsonl(path):
        sentence = (row.get("sentence") or "").strip()
        if not sentence:
            continue
        records.append({
            "sentence_id": str(row["sentence_id"]),
            "row_id": int(row["row_id"]),
            "domain": row.get("domain", ""),
            "category": row.get("category", ""),
            "word": row.get("word", ""),
            "english_sentence": sentence,
        })
    return records


# --- stage 1+2: translate, then rule-check locally ---------------------------

def _apply_tier1_translations(
    records: list[dict],
    translations: dict[str, str],
    config: dict,
    state: PipelineState,
) -> None:
    """Record one job's worth of Tier-1 results. A sentence the model dropped
    from its chunk is recorded as a failed attempt, never as an empty
    translation, so it escalates to Tier 2 instead of being exported blank."""
    max_attempts = config["pipeline"]["max_attempts_per_sentence"]
    tier1_cfg = config["translation"]["tier1"]
    for record in records:
        sid = record["sentence_id"]
        state.record_attempt(sid, tier=1)
        telugu = translations.get(sid)
        if telugu:
            state.set_translation(sid, telugu, tier=1,
                                  provider=tier1_cfg["provider"], model=tier1_cfg["model"])
        else:
            state.mark_failed_attempt(sid, "tier-1 translation unavailable", max_attempts)


def _guard_submission_failure(exc, failures: int, limit: int, stage: str) -> None:
    """Stop a submission loop that cannot succeed.

    Both batch stages requeue a failed submission and sleep, which is right for
    a transient blip and catastrophic otherwise: with no ceiling, a dead API key
    or an exhausted balance spins forever at one doomed call every 30 seconds.
    Submission does not go through call_with_backoff, so this is the only place
    that ceiling can live.
    """
    if retry.is_permanent(exc):
        raise RuntimeError(f"{stage} batch submission failed permanently: {exc}") from exc
    if failures + 1 >= limit:
        raise RuntimeError(
            f"{stage} batch submission failed {failures + 1} times in a row, giving up: {exc}"
        ) from exc


def _translate_tier1(
    records: list[dict],
    config: dict,
    state: PipelineState,
    prompts_dir: str,
) -> None:
    """Translate the whole-corpus pass via Gemini Batch Mode, in sub-jobs with a
    pool in flight.

    Deliberately the same shape as _judge_records below: both async stages
    submit, checkpoint the job name, poll, and apply results identically, so an
    interruption behaves the same way whichever stage it lands in.
    """
    if not records:
        return
    tier1_cfg = config["translation"]["tier1"]
    chunks = list(batching.make_batches(records, tier1_cfg["chunk_size"]))
    max_chunks = tier1_cfg["batch_job_max_chunks"]
    max_concurrent = tier1_cfg["max_concurrent_jobs"]
    queue = [chunks[i : i + max_chunks] for i in range(0, len(chunks), max_chunks)]
    logger.info(
        "tier-1 translating %d sentence(s) in %d chunk(s) across %d job(s)",
        len(records), len(chunks), len(queue),
    )

    max_submit_failures = config.get("retry", {}).get("max_api_attempts", 4)
    submit_failures = 0
    in_flight: list[tuple[str, list[list[dict]]]] = []
    while queue or in_flight:
        while queue and len(in_flight) < max_concurrent:
            job_chunks = queue.pop(0)
            try:
                job_name = gemini_translator.submit_translation_batch_job(
                    job_chunks, config, prompts_dir
                )
            except Exception as exc:  # noqa: BLE001 - requeue rather than lose the work
                _guard_submission_failure(exc, submit_failures, max_submit_failures, "tier-1")
                submit_failures += 1
                logger.error("tier-1 batch submission failed (%d/%d), requeuing: %s",
                             submit_failures, max_submit_failures, exc)
                queue.insert(0, job_chunks)
                time.sleep(30)
                break
            submit_failures = 0
            state.add_pending_translation_batch(job_name, job_chunks)
            state.save()
            in_flight.append((job_name, job_chunks))

        if not in_flight:
            continue

        time.sleep(30)
        still_running: list[tuple[str, list[list[dict]]]] = []
        for job_name, job_chunks in in_flight:
            try:
                job = gemini_translator.check_batch_status(job_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not check tier-1 batch %s: %s", job_name, exc)
                still_running.append((job_name, job_chunks))
                continue
            if not gemini_translator.is_terminal(job):
                still_running.append((job_name, job_chunks))
                continue

            flat = [r for chunk in job_chunks for r in chunk]
            if gemini_translator.is_batch_usable(job):
                translations = gemini_translator.fetch_translation_batch_results(job, job_chunks)
            else:
                logger.error("tier-1 batch %s ended in state %s", job_name, job.state)
                translations = {}
            _apply_tier1_translations(flat, translations, config, state)
            state.remove_pending_translation_batch(job_name)
            state.save()
        in_flight = still_running


def _reconnect_pending_translation_batches(config: dict, state: PipelineState) -> None:
    """Rejoin any Tier-1 job submitted (and paid for) before an interruption."""
    pending = state.get_pending_translation_batches()
    if not pending:
        return
    logger.info("reconnecting to %d in-flight tier-1 batch job(s)", len(pending))
    for batch in pending:
        job_name = batch["job_name"]
        job_chunks = batch["chunks"]
        try:
            job = gemini_translator.poll_translation_batch_job(job_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("could not reconnect to tier-1 batch %s: %s", job_name, exc)
            state.remove_pending_translation_batch(job_name)
            state.save()
            continue
        flat = [r for chunk in job_chunks for r in chunk]
        translations = (
            gemini_translator.fetch_translation_batch_results(job, job_chunks)
            if gemini_translator.is_batch_usable(job)
            else {}
        )
        _apply_tier1_translations(flat, translations, config, state)
        state.remove_pending_translation_batch(job_name)
        state.save()


def _translate_tier2(records: list[dict], config: dict, state: PipelineState, prompts_dir: str) -> None:
    """Escalate failed sentences to a stronger Gemini model than Tier 1 used,
    with the domain/category/word context in the prompt."""
    if not records:
        return
    tier2_cfg = config["translation"]["tier2"]
    chunk_size = tier2_cfg["chunk_size"]
    max_attempts = config["pipeline"]["max_attempts_per_sentence"]
    for chunk in batching.make_batches(records, chunk_size):
        try:
            translations = gemini_translator.translate_batch(chunk, config, prompts_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("tier-2 translation chunk failed: %s", exc)
            translations = {}
        for record in chunk:
            sid = record["sentence_id"]
            state.record_attempt(sid, tier=2)
            telugu = translations.get(sid)
            if telugu:
                state.set_translation(sid, telugu, tier=2,
                                      provider=tier2_cfg["provider"], model=tier2_cfg["model"])
            else:
                state.mark_failed_attempt(sid, "tier-2 translation unavailable", max_attempts)


def _apply_rule_checks(
    records: list[dict],
    config: dict,
    state: PipelineState,
    paths: dict,
) -> list[dict]:
    """Local, free gate applied before anything costs a judge call. Returns the
    records whose translations passed and are ready to be judged."""
    rules = config["rules"]
    max_attempts = config["pipeline"]["max_attempts_per_sentence"]
    ready: list[dict] = []
    rejected: list[dict] = []

    for record in records:
        entry = state.get(record["sentence_id"])
        if not entry or entry["status"] != "translated" or not entry["telugu_sentence"]:
            continue
        passed, reason = validation.check_translation_rules(
            entry["english_sentence"], entry["telugu_sentence"], rules
        )
        if passed:
            ready.append({**record, "telugu_sentence": entry["telugu_sentence"]})
        else:
            rejected.append({
                "sentence_id": record["sentence_id"],
                "english_sentence": entry["english_sentence"],
                "telugu_sentence": entry["telugu_sentence"],
                "tier": entry["tier"],
                "stage": "rules",
                "reason": reason,
            })
            state.mark_failed_attempt(record["sentence_id"], f"rule check: {reason}", max_attempts)

    if rejected:
        jsonl_storage.append_jsonl(os.path.join(paths["data_rejected"], "rejected.jsonl"), rejected)
        logger.info("rule checks rejected %d/%d translation(s)", len(rejected), len(records))
    return ready


# --- stage 3: near-duplicate check among a word's own translations -----------

def _filter_near_duplicates(
    records: list[dict],
    config: dict,
    state: PipelineState,
    paths: dict,
) -> list[dict]:
    """Rejects a translation that is near-identical to one already accepted for
    another sentence of the SAME word. The embedding module is the only
    authority here; the judge is never asked about duplicates."""
    if not records:
        return []

    emb_cfg = config["embedding"]
    threshold = config["rules"]["near_duplicate_similarity_threshold"]
    max_attempts = config["pipeline"]["max_attempts_per_sentence"]

    candidate_vecs = similarity.embed_translations(
        [r["telugu_sentence"] for r in records],
        model=emb_cfg["model"],
        passage_prefix=emb_cfg.get("passage_prefix", ""),
    )

    # Existing accepted translations, grouped per word, embedded once per batch.
    row_ids = {r["row_id"] for r in records}
    existing_by_row: dict[int, list[list[float]]] = {}
    for row_id in row_ids:
        existing = state.accepted_translations_for_row(row_id)
        existing_by_row[row_id] = (
            similarity.embed_translations(
                existing, model=emb_cfg["model"], passage_prefix=emb_cfg.get("passage_prefix", "")
            )
            if existing
            else []
        )

    kept: list[dict] = []
    rejected: list[dict] = []
    for record, vec in zip(records, candidate_vecs):
        pool = existing_by_row[record["row_id"]]
        if similarity.find_near_duplicate(vec, pool, threshold) is not None:
            rejected.append({
                "sentence_id": record["sentence_id"],
                "english_sentence": record["english_sentence"],
                "telugu_sentence": record["telugu_sentence"],
                "stage": "near_duplicate",
                "reason": f"near-duplicate of another accepted translation for word {record['word']!r}",
            })
            state.mark_failed_attempt(record["sentence_id"], "near-duplicate translation", max_attempts)
            continue
        pool.append(vec)
        kept.append(record)

    if rejected:
        jsonl_storage.append_jsonl(os.path.join(paths["data_rejected"], "rejected.jsonl"), rejected)
        logger.info("near-duplicate check rejected %d translation(s)", len(rejected))
    return kept


# --- stage 4+5+6: judge, semantic similarity, final QC -----------------------

def _apply_judge_results(
    records: list[dict],
    evaluations: dict[str, dict],
    config: dict,
    state: PipelineState,
    paths: dict,
) -> None:
    """Recomputes the accept/reject decision locally from the judge's
    per-dimension scores plus the cross-lingual similarity, then records it."""
    if not records:
        return

    emb_cfg = config["embedding"]
    thresholds = config["judge_thresholds"]
    semantic_threshold = config["semantic_similarity_threshold"]
    max_attempts = config["pipeline"]["max_attempts_per_sentence"]

    judged = [r for r in records if r["sentence_id"] in evaluations]
    unjudged = [r for r in records if r["sentence_id"] not in evaluations]
    for record in unjudged:
        state.mark_failed_attempt(record["sentence_id"], "judge returned no evaluation", max_attempts)

    if not judged:
        return

    sims = similarity.semantic_similarity_scores(
        [(r["english_sentence"], r["telugu_sentence"]) for r in judged],
        model=emb_cfg["model"],
        query_prefix=emb_cfg.get("query_prefix", ""),
        passage_prefix=emb_cfg.get("passage_prefix", ""),
    )

    rejected: list[dict] = []
    accepted_count = 0
    missing_romanization = 0
    for record, semantic_similarity in zip(judged, sims):
        sid = record["sentence_id"]
        evaluation = evaluations[sid]
        decision, overall_score, reason = validation.compute_translation_decision(
            evaluation, thresholds, semantic_similarity, semantic_threshold
        )
        if decision == "PASS":
            # The judge returns the Roman-script rendering alongside its scores,
            # so romanization costs no extra API call. It is a deliverable, not
            # a quality signal -- a missing one is logged, never a rejection.
            romanized = (evaluation.get("telugu_romanized") or "").strip() or None
            if romanized is None:
                missing_romanization += 1
            state.accept(
                sid, evaluation, overall_score, semantic_similarity,
                telugu_romanized=romanized, judge_model=config["judge"]["model"],
            )
            accepted_count += 1
        else:
            rejected.append({
                "sentence_id": sid,
                "english_sentence": record["english_sentence"],
                "telugu_sentence": record["telugu_sentence"],
                "stage": "judge",
                "semantic_similarity": semantic_similarity,
                "judge_decision": evaluation.get("decision"),
                "reason": reason,
            })
            state.mark_failed_attempt(sid, reason, max_attempts)

    if rejected:
        jsonl_storage.append_jsonl(os.path.join(paths["data_rejected"], "rejected.jsonl"), rejected)
    logger.info("judged %d translation(s): %d accepted, %d rejected", len(judged), accepted_count, len(rejected))
    if missing_romanization:
        logger.warning(
            "%d accepted translation(s) came back without a romanization", missing_romanization
        )


def _judge_records(records: list[dict], config: dict, state: PipelineState, paths: dict, prompts_dir: str) -> None:
    """Judges via Gemini Batch Mode, in sub-jobs small enough that no single job
    is oversized, with a small pool in flight at once."""
    if not records:
        return

    judge_cfg = config["judge"]
    chunks = list(batching.make_batches(records, judge_cfg["chunk_size"]))
    max_chunks = judge_cfg["batch_job_max_chunks"]
    max_concurrent = judge_cfg["max_concurrent_jobs"]
    queue = [chunks[i : i + max_chunks] for i in range(0, len(chunks), max_chunks)]
    logger.info("judging %d translation(s) in %d chunk(s) across %d job(s)", len(records), len(chunks), len(queue))

    max_submit_failures = config.get("retry", {}).get("max_api_attempts", 4)
    submit_failures = 0
    in_flight: list[tuple[str, list[list[dict]]]] = []
    while queue or in_flight:
        while queue and len(in_flight) < max_concurrent:
            job_chunks = queue.pop(0)
            try:
                job_name = judge_agent.submit_judge_batch_job(job_chunks, config, prompts_dir)
            except Exception as exc:  # noqa: BLE001 - requeue rather than lose the work
                _guard_submission_failure(exc, submit_failures, max_submit_failures, "judge")
                submit_failures += 1
                logger.error("judge batch submission failed (%d/%d), requeuing: %s",
                             submit_failures, max_submit_failures, exc)
                queue.insert(0, job_chunks)
                time.sleep(30)
                break
            submit_failures = 0
            state.add_pending_judge_batch(job_name, job_chunks)
            state.save()
            in_flight.append((job_name, job_chunks))

        if not in_flight:
            continue

        time.sleep(30)
        still_running: list[tuple[str, list[list[dict]]]] = []
        for job_name, job_chunks in in_flight:
            try:
                job = judge_agent.check_batch_status(job_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not check judge batch %s: %s", job_name, exc)
                still_running.append((job_name, job_chunks))
                continue
            if not judge_agent.is_terminal(job):
                still_running.append((job_name, job_chunks))
                continue

            flat = [r for chunk in job_chunks for r in chunk]
            if judge_agent.is_batch_usable(job):
                evaluations = judge_agent.fetch_judge_batch_results(job, job_chunks)
            else:
                logger.error("judge batch %s ended in state %s", job_name, job.state)
                evaluations = {}
            _apply_judge_results(flat, evaluations, config, state, paths)
            state.remove_pending_judge_batch(job_name)
            state.save()
        in_flight = still_running


def _reconnect_pending_judge_batches(config: dict, state: PipelineState, paths: dict) -> None:
    """Rejoins any judge job that was already submitted (and paid for) before a
    previous run was interrupted, rather than resubmitting the same work."""
    pending = state.get_pending_judge_batches()
    if not pending:
        return
    logger.info("reconnecting to %d in-flight judge batch job(s)", len(pending))
    for batch in pending:
        job_name = batch["job_name"]
        job_chunks = batch["chunks"]
        try:
            job = judge_agent.poll_judge_batch_job(job_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("could not reconnect to judge batch %s: %s", job_name, exc)
            state.remove_pending_judge_batch(job_name)
            state.save()
            continue
        flat = [r for chunk in job_chunks for r in chunk]
        evaluations = judge_agent.fetch_judge_batch_results(job, job_chunks) if judge_agent.is_batch_usable(job) else {}
        _apply_judge_results(flat, evaluations, config, state, paths)
        state.remove_pending_judge_batch(job_name)
        state.save()


# --- export ------------------------------------------------------------------

def export_final(state: PipelineState, config: dict, paths: dict) -> None:
    tier1_cfg = config["translation"]["tier1"]
    judge_cfg = config["judge"]

    records = []
    for entry in state.all_entries():
        if entry["status"] != "complete":
            continue
        scores = entry.get("scores") or {}
        records.append({
            "sentence_id": entry["sentence_id"],
            "row_id": entry["row_id"],
            "domain": entry["domain"],
            "category": entry["category"],
            "word": entry["word"],
            "source_language": tier1_cfg["source_language_code"],
            "target_language": tier1_cfg["target_language_code"],
            "english_sentence": entry["english_sentence"],
            "telugu_sentence": entry["telugu_sentence"],
            "telugu_romanized": entry.get("telugu_romanized"),
            # Recorded at translation/judging time. Deriving these from the
            # current config instead would relabel every sentence whenever a
            # model changes, making the corpus claim a provenance that is false.
            "translation_provider": entry.get("translation_provider"),
            "translation_model": entry.get("translation_model"),
            "judge_provider": judge_cfg["provider"],
            "judge_model": entry.get("judge_model"),
            **{k: scores.get(k) for k in validation.DIMENSION_KEYS},
            "overall_score": entry["overall_score"],
            "semantic_similarity": entry["semantic_similarity"],
            "status": "accepted",
        })

    os.makedirs(paths["data_final"], exist_ok=True)
    jsonl_storage.write_jsonl(os.path.join(paths["data_final"], "final_translations.jsonl"), records)
    excel_storage.write_final_excel(os.path.join(paths["data_final"], "final_translations.xlsx"), records)
    logger.info("exported %d accepted translation(s) to %s", len(records), paths["data_final"])


# --- entrypoint ---------------------------------------------------------------

def run(config_path: str = "config/config.yaml", limit: int | None = None) -> None:
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
    config = load_config(config_path)
    setup_logging(config)
    paths = config["paths"]
    prompts_dir = paths["prompts_dir"]

    for key in ("data_raw", "data_processed", "data_rejected", "data_final"):
        os.makedirs(paths[key], exist_ok=True)

    logger.info("loading English sentences from %s", paths["input_file"])
    all_records = load_input_sentences(paths["input_file"])
    if limit:
        all_records = all_records[:limit]
        logger.info("limit=%d -- running against the first %d sentence(s) only", limit, len(all_records))
    logger.info("loaded %d sentence(s)", len(all_records))

    state = PipelineState(paths["state_file"])
    _reconnect_pending_translation_batches(config, state)
    _reconnect_pending_judge_batches(config, state, paths)

    max_attempts = config["pipeline"]["max_attempts_per_sentence"]

    for attempt_round in range(1, max_attempts + 1):
        todo = state.sentences_needing_work(all_records)
        if not todo:
            break
        tier = 1 if attempt_round == 1 else 2
        logger.info("round %d/%d (tier %d): %d sentence(s) to process", attempt_round, max_attempts, tier, len(todo))

        for batch_index, batch in enumerate(batching.make_batches(todo, config["pipeline"]["batch_size"]), start=1):
            needs_translation = [
                r for r in batch
                if (entry := state.get(r["sentence_id"])) is None or entry["status"] == "pending"
            ]
            if tier == 1:
                _translate_tier1(needs_translation, config, state, prompts_dir)
            else:
                _translate_tier2(needs_translation, config, state, prompts_dir)
            state.save()

            ready = _apply_rule_checks(batch, config, state, paths)
            ready = _filter_near_duplicates(ready, config, state, paths)
            state.save()

            _judge_records(ready, config, state, paths, prompts_dir)
            state.save()
            logger.info(
                "round %d: batch %d done -- %s", attempt_round, batch_index, state.status_counts()
            )

    export_final(state, config, paths)
    logger.info("pipeline complete: %s", state.status_counts())
