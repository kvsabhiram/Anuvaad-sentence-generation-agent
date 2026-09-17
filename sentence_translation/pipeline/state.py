"""Persistent run state so a 158k-sentence pipeline can be interrupted and
resumed without re-translating or re-judging work that already succeeded.

Keyed by sentence_id (e.g. "4_1"). Embeddings are deliberately NOT persisted --
they would balloon this file at 158k rows; the orchestrator recomputes the few
it needs per batch, which is cheap on a local GPU.

Statuses:
  pending      -- not yet translated
  translated   -- has a Telugu candidate that passed the local rule checks
  complete     -- judged, passed every threshold, accepted
  needs_review -- exhausted its attempts (Tier 1 then Tier 2) without passing
"""

from __future__ import annotations

import json
import os
from typing import Any


class PipelineState:
    def __init__(self, path: str):
        self.path = path
        self._data: dict[str, Any] = {"sentences": {}}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        self._data.setdefault("sentences", {})
        self._data.setdefault("pending_judge_batches", [])
        self._data.setdefault("pending_translation_batches", [])

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    def ensure_sentence(self, record: dict) -> dict:
        key = str(record["sentence_id"])
        if key not in self._data["sentences"]:
            self._data["sentences"][key] = {
                "sentence_id": record["sentence_id"],
                "row_id": record["row_id"],
                "domain": record["domain"],
                "category": record["category"],
                "word": record["word"],
                "english_sentence": record["english_sentence"],
                "status": "pending",
                "attempts": 0,
                "tier": None,
                "telugu_sentence": None,
                "telugu_romanized": None,
                # Recorded when the translation/judgement actually happens, not
                # read from config at export time -- otherwise changing a model
                # in config retroactively relabels sentences produced by the
                # old one, and the corpus claims a provenance that is false.
                "translation_provider": None,
                "translation_model": None,
                "judge_model": None,
                "scores": None,
                "semantic_similarity": None,
                "overall_score": None,
                "reason": "",
            }
        return self._data["sentences"][key]

    def get(self, sentence_id: str) -> dict | None:
        return self._data["sentences"].get(str(sentence_id))

    def sentences_needing_work(self, all_records: list[dict]) -> list[dict]:
        """Records not yet accepted and not yet given up on."""
        out = []
        for record in all_records:
            entry = self.ensure_sentence(record)
            if entry["status"] in ("pending", "translated"):
                out.append(record)
        return out

    def record_attempt(self, sentence_id: str, tier: int) -> None:
        entry = self._data["sentences"][str(sentence_id)]
        entry["attempts"] += 1
        entry["tier"] = tier

    def set_translation(
        self,
        sentence_id: str,
        telugu_sentence: str,
        tier: int,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        entry = self._data["sentences"][str(sentence_id)]
        entry["telugu_sentence"] = telugu_sentence
        entry["tier"] = tier
        entry["translation_provider"] = provider
        entry["translation_model"] = model
        entry["status"] = "translated"

    def accept(
        self,
        sentence_id: str,
        scores: dict,
        overall_score: float,
        semantic_similarity: float,
        telugu_romanized: str | None = None,
        judge_model: str | None = None,
    ) -> None:
        entry = self._data["sentences"][str(sentence_id)]
        entry["scores"] = scores
        entry["overall_score"] = overall_score
        entry["semantic_similarity"] = semantic_similarity
        entry["judge_model"] = judge_model
        # Romanization rides along with the judge response. It is a deliverable,
        # not a quality signal: a missing one is logged by the caller but never
        # blocks acceptance of an otherwise-good translation.
        entry["telugu_romanized"] = telugu_romanized
        entry["status"] = "complete"
        entry["reason"] = ""

    def mark_failed_attempt(self, sentence_id: str, reason: str, max_attempts: int) -> None:
        """Record why this attempt failed. The sentence goes back to pending for
        a Tier-2 escalation, or to needs_review once attempts are exhausted."""
        entry = self._data["sentences"][str(sentence_id)]
        entry["reason"] = reason
        if entry["attempts"] >= max_attempts:
            entry["status"] = "needs_review"
        else:
            entry["status"] = "pending"

    def accepted_translations_for_row(self, row_id: int) -> list[str]:
        """Every accepted Telugu translation belonging to one word (row_id), for
        near-duplicate checking among that word's sibling sentences."""
        return [
            e["telugu_sentence"]
            for e in self._data["sentences"].values()
            if e["row_id"] == row_id and e["status"] == "complete" and e["telugu_sentence"]
        ]

    def all_entries(self) -> list[dict]:
        return list(self._data["sentences"].values())

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._data["sentences"].values():
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        return counts

    # --- in-flight judge batch jobs, so a restart reconnects instead of
    # --- resubmitting work that is already queued and paid for
    def add_pending_judge_batch(self, job_name: str, chunks: list[list[dict]]) -> None:
        self._data["pending_judge_batches"].append({"job_name": job_name, "chunks": chunks})

    def get_pending_judge_batches(self) -> list[dict]:
        return list(self._data.get("pending_judge_batches", []))

    def remove_pending_judge_batch(self, job_name: str) -> None:
        self._data["pending_judge_batches"] = [
            b for b in self._data.get("pending_judge_batches", []) if b["job_name"] != job_name
        ]

    # --- in-flight Tier-1 translation batch jobs. Same reconnect guarantee as
    # --- the judge jobs above: a submitted job is already costing money, so a
    # --- restart must rejoin it rather than pay to translate the same chunk twice.
    def add_pending_translation_batch(self, job_name: str, chunks: list[list[dict]]) -> None:
        self._data["pending_translation_batches"].append({"job_name": job_name, "chunks": chunks})

    def get_pending_translation_batches(self) -> list[dict]:
        return list(self._data.get("pending_translation_batches", []))

    def remove_pending_translation_batch(self, job_name: str) -> None:
        self._data["pending_translation_batches"] = [
            b for b in self._data.get("pending_translation_batches", []) if b["job_name"] != job_name
        ]
