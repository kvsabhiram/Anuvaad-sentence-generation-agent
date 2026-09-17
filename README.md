# Anuvaad Forge

An end-to-end factory for building an English→Telugu machine translation
training corpus: generate high-quality English sentences for a vocabulary list,
then translate and quality-gate them into a parallel corpus.

Two pipelines, run in sequence. Each is self-contained (own venv, config, data,
prompts) and independently resumable.

```
  input vocabulary (15,490 domain/category/word rows)
            │
            ▼
  ┌─────────────────────────┐
  │  sentence_generation/   │   generate 4-5 English sentences per word,
  │                         │   LLM-judged, embedding-deduped
  └───────────┬─────────────┘
              │  158,490 accepted English sentences
              ▼
  ┌─────────────────────────┐
  │  sentence_translation/  │   translate to Telugu (native + romanized),
  │                         │   rule-checked, LLM-judged, QC-gated
  └───────────┬─────────────┘
              ▼
     English↔Telugu parallel corpus
```

## Phase 1 — `sentence_generation/`

**Status: complete.** 158,490 accepted sentences across 15,370 of 15,490 words
(99.2% word coverage).

Generates natural English sentences that use each target word correctly, then
filters them through an LLM judge with deterministic per-dimension thresholds
and local embedding-based near-duplicate rejection. Provider-switchable between
Gemini and OpenAI for both generation and judging.

Output: `sentence_generation/data/final/final_sentences.{jsonl,xlsx}`
Full write-up: `sentence_generation/PIPELINE_RECORD.txt`

## Phase 2 — `sentence_translation/`

**Status: built and piloted; full 158,490-sentence run pending.**

Translates Phase 1's output into Telugu with Gemini Flash-Lite (Batch Mode),
gates every translation through local deterministic rules (script, leakage,
length, number preservation, near-duplicates), judges it with a different,
stronger Gemini model across 7 dimensions, and independently verifies it with a
local cross-lingual embedding check. Sentences that fail escalate to that
stronger model rather than re-asking the one that just failed.

Output: `sentence_translation/data/final/final_translations.{jsonl,xlsx}` —
each accepted sentence carries the Telugu in **both native script and
romanized** form.

Details: `sentence_translation/README.md`

## Design rules shared by both pipelines

These are the conventions both phases follow, each learned from a failure:

- **No LLM grades its own homework.** Judges return per-dimension scores; the
  PASS/FAIL gate is recomputed locally from thresholds in config. A judge that
  says "PASS" on failing scores is overruled.
- **Duplicate detection belongs to the embedding model,** never the judge.
- **State is checkpointed after every phase,** atomically, so multi-hour runs
  survive interruption and resume without repeating paid work.
- **API constraints get verified before the code is written,** not discovered
  mid-run.
- **Thresholds are calibrated against measured data,** not guessed, and the
  measurements are recorded in the config comments next to the value.
