# Sentence Translation Agent (English → Telugu)

Translates the 158,490 English sentences produced by the sentence-generation
pipeline into Telugu, judges each translation, and exports only the ones that
pass every check.

## Flow

```
input/english_sentences.jsonl  (158,490 sentences)
        │
        ▼
  Tier 1 translate  (gemini-3.5-flash-lite, Batch Mode)
        │
        ▼
  Deterministic rules   (local, free — no API cost)
    empty · Telugu script · English leakage · length anomaly
    · number preservation · near-duplicate
        │ FAIL ──────────────────► Tier 2: gemini-3.7-flash escalation
        ▼ PASS                            │
  Judge  (gemini-3.7-flash, thinking_level=LOW,  ◄┘
          7 dimensions + romanization, Batch Mode)
        │
        ▼
  Local cross-lingual embedding  (multilingual-e5-base)
        │
        ▼
  Final QC  (thresholds recomputed locally, judge's own verdict ignored)
        │
   PASS ─┴─ FAIL → retry once via Tier 2, then needs_review
        ▼
  data/final/final_translations.{jsonl,xlsx}
     native Telugu + romanized, per sentence
```

Three models, and the separation is deliberate: the Tier-1 translator, the
Tier-2 escalation translator and the judge are all different. A model grading
its own output is not an independent gate, and escalating to the same model
that just failed at temperature 0 returns the same failing answer.

## Setup

```bash
cd sentence_translation
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

Then put the key in `.env`:

```
GEMINI_API_KEY=...
```

## Running

Always pilot first — it exercises every stage end to end for a few rupees:

```bash
.venv/bin/python3 main.py --limit 25
```

Full run (resumable; safe to interrupt and restart):

```bash
nohup .venv/bin/python3 main.py > logs/run.log 2>&1 < /dev/null &
disown
```

Progress:

```bash
tail -f logs/run.log
.venv/bin/python3 -c "
import json; from collections import Counter
d = json.load(open('data/state.json'))
print(Counter(v['status'] for v in d['sentences'].values()))"
```

## Things worth knowing before you run it

**Cost and tokens for a full pass**, measured per-sentence from live API calls
against a 384-sentence stratified sample (24 domains, 375 distinct words) and
scaled to the corpus. Assumes ~11% of sentences escalate to Tier 2:

| stage | model | tokens | cost |
|-------|-------|--------|------|
| Tier-1 translate | flash-lite, Batch | 23.7M | $14.05 |
| Judge | 3.7-flash LOW, Batch | 59.6M | $68.93 |
| Tier-2 escalate | 3.7-flash, sync | 3.1M | $6.89 |
| **total** | | **86.4M** | **$89.87 ≈ ₹7,900** |

The judge is ~69% of all tokens — scoring the corpus costs more than twice what
producing it does. Cost is roughly linear in the escalation rate at ~$1.25 per
percentage point, so being wrong about that rate moves the bill by a few hundred
rupees, not a few thousand.

One thing would change this materially: leaving `judge.thinking_level` unset
reverts the judge to its MEDIUM default and adds **$9–33** for no measured
quality gain.

**Runtime is set by batch job count, not by tokens.** Batch jobs carry roughly
60–90s of fixed overhead each regardless of size, and the orchestrator translates
a whole work unit before blocking on judging it. `pipeline.batch_size` therefore
controls how many times the run pays that serialization penalty: at the old 200
it was 793 sequential round trips, at 20,000 it is 8. Concurrency is bounded by
the ~3M enqueued-token quota across active jobs rather than the 100-concurrent-job
limit — see the arithmetic in `config/config.yaml`.

Google publishes a 24h *target* for batch completion, not a guarantee, and jobs
hard-expire at 48h with results unrecoverable. Plan for a run measured in days,
and pilot before trusting any schedule: our own latency data is n=2.

**Retries escalate rather than repeat.** Re-asking one model the same question
at temperature 0 returns the same failing answer, so a failed sentence goes to a
*different, stronger* model rather than looping on the one that just failed.
Keep `translation.tier2.model` different from `translation.tier1.model`, and the
judge different from both, or these stages stop being independent.

**Translation quality against a dedicated Indic MT engine is unsettled.** A
25-sentence pilot put Sarvam's Telugu ahead of Gemini's; a 384-sentence
stratified run put Gemini ahead; the controlled head-to-head that would settle
it was never completed. The pipeline now runs fully on Gemini — if translation
quality disappoints in production, this is the first assumption to re-examine.

**What the semantic similarity check can and cannot do.** Measured against this
exact model: correct translations score 0.89–0.92, unrelated sentences 0.73–0.74,
but a *flipped negation* still scores 0.85. So the 0.80 threshold catches gross
failures (wrong language, unrelated output, garbage) and nothing subtler.
Meaning, completeness and hallucination are the judge's job. Raising this
threshold to try to catch semantic errors will only start rejecting good
translations.

**GPU is optional.** Embedding the whole corpus takes ~2.6 min on GPU vs ~21 min
on CPU — irrelevant next to a multi-day batch run. If `torch.cuda.is_available()`
is False it just runs on CPU.

**Romanized output rides along with the judge.** Every accepted sentence is
exported in both native Telugu script (`telugu_sentence`) and Roman script
(`telugu_romanized`). The Gemini judge returns the romanization in the same
response as its scores, so this costs no extra API call, no extra rate-limit
budget, and no extra translation spend — the alternative, a dedicated
transliteration endpoint, would have meant a second paid call per sentence.

Romanization is treated as a deliverable, not a quality signal: if the judge
omits it, the sentence is still accepted on its scores and the omission is
logged as a warning. It never causes a rejection.

## Layout

| path | role |
|------|------|
| `agents/translator_gemini.py` | Tier 1 (Batch Mode) and Tier 2 (sync) translation |
| `agents/judge_translation_gemini.py` | Judge + romanization (Batch Mode + sync) |
| `pipeline/validation.py` | Deterministic rules + final QC recompute |
| `pipeline/orchestrator.py` | Stage wiring, batching, resume, export |
| `pipeline/state.py` | Resumable per-sentence state (`data/state.json`) |
| `embedding/` | Local cross-lingual embeddings + similarity |
| `prompts/judge_translation.txt` | Judge prompt |
| `prompts/translator.txt` | Translation prompt, shared by Tier 1 and Tier 2 |
| `config/config.yaml` | All tunables, with rationale in comments |
