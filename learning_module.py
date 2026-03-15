# learning_module.py
# Save logs and update local learning data.

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class LearningConfig:
    """Settings for local learning."""
    # Minimum score for KB writeback
    min_answer_quality_score: int = 85

    # Maximum KB writeback length
    max_kb_writeback_chars: int = 4000

    # Whether to always save eval items
    always_add_to_evalset: bool = False

    # Require reference answer for evalset
    add_to_evalset_requires_reference: bool = True


class LearningModule:
    """Handles logs and local updates."""

    def __init__(
        self,
        storage_dir: str = "storage",
        *,
        logs_file: str = "logs.jsonl",
        error_patterns_file: str = "error_patterns.json",
        evalset_file: str = "evalset.jsonl",
        versions_file: str = "versions.json",
        config: Optional[LearningConfig] = None,
    ) -> None:
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.logs_path = os.path.join(self.storage_dir, logs_file)
        self.error_patterns_path = os.path.join(self.storage_dir, error_patterns_file)
        self.evalset_path = os.path.join(self.storage_dir, evalset_file)
        self.versions_path = os.path.join(self.storage_dir, versions_file)

        self.cfg = config or LearningConfig()

        # Initialize files if missing
        if not os.path.exists(self.error_patterns_path):
            self._write_json(self.error_patterns_path, {"tag_counts": {}, "updated_at": self._now_ts()})
        if not os.path.exists(self.versions_path):
            self._write_json(
                self.versions_path,
                {
                    "prompt_version": "v1",
                    "rubric_version": "v1",
                    "retrieval_policy_version": "v1",
                    "changelog": [],
                    "updated_at": self._now_ts(),
                },
            )

    # -------------------------------
    # Logging
    # -------------------------------

    def log_interaction(self, record: Dict[str, Any]) -> str:
        """Save one interaction log."""

        if not isinstance(record, dict):
            raise ValueError("log_interaction: record must be a dict.")

        interaction_id = f"it_{uuid.uuid4().hex}"
        record_out = dict(record)
        record_out["interaction_id"] = interaction_id
        record_out["ts"] = self._now_ts()

        with open(self.logs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_out, ensure_ascii=False) + "\n")

        return interaction_id

    # -------------------------------
    # Error pattern library
    # -------------------------------

    def update_error_patterns(self, error_tags: List[str]) -> None:
        """Update error statistics."""

        if not error_tags:
            return

        data = self._read_json(self.error_patterns_path)
        tag_counts = data.get("tag_counts", {})
        if not isinstance(tag_counts, dict):
            tag_counts = {}

        for t in error_tags:
            t = str(t).strip()
            if not t:
                continue
            tag_counts[t] = int(tag_counts.get(t, 0)) + 1

        data["tag_counts"] = tag_counts
        data["updated_at"] = self._now_ts()
        self._write_json(self.error_patterns_path, data)

    # -------------------------------
    # KB write-back (safe self-learning)
    # -------------------------------

    def maybe_writeback_to_kb(
        self,
        *,
        kb_ingest_text_fn,
        problem_text: str,
        chosen_answer_text: str,
        chosen_score_total: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save good answers into the knowledge base."""

        if not isinstance(problem_text, str) or not problem_text.strip():
            return False
        if not isinstance(chosen_answer_text, str) or not chosen_answer_text.strip():
            return False

        score = int(chosen_score_total)
        if score < self.cfg.min_answer_quality_score:
            return False

        # Build a short "knowledge card" text for future retrieval
        card = self._build_kb_card(problem_text.strip(), chosen_answer_text.strip())

        # Limit size
        if len(card) > self.cfg.max_kb_writeback_chars:
            card = card[: self.cfg.max_kb_writeback_chars].rstrip() + "\n...(truncated)"

        meta = metadata.copy() if metadata else {}
        meta.update({
            "source": "self_learned",
            "type": "solution_card",
            "quality_score": score,
        })

        # Ingest into KB through RAG module function
        kb_ingest_text_fn(card, metadata=meta)
        return True

    def _build_kb_card(self, problem_text: str, answer_text: str) -> str:
        """
        Create a compact KB entry that is useful for retrieval.
        """
        return f"""[KB_CARD]
QUESTION:
{problem_text}

SOLUTION:
{answer_text}
"""

    # -------------------------------
    # Eval set management
    # -------------------------------

    def maybe_add_to_evalset(
        self,
        *,
        problem_text: str,
        reference_answer: Optional[str],
        chosen_answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Decide whether to add an item into evalset.jsonl for future regression tests.
        """
        if not isinstance(problem_text, str) or not problem_text.strip():
            return False
        if not isinstance(chosen_answer, str) or not chosen_answer.strip():
            return False

        ref = (reference_answer or "").strip()

        if not self.cfg.always_add_to_evalset:
            if self.cfg.add_to_evalset_requires_reference and not ref:
                return False

        item = {
            "eval_id": f"ev_{uuid.uuid4().hex}",
            "ts": self._now_ts(),
            "problem_text": problem_text.strip(),
            "reference_answer": ref,
            "chosen_answer": chosen_answer.strip(),
            "metadata": metadata or {},
        }

        with open(self.evalset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return True

    # -------------------------------
    # Version management
    # -------------------------------

    def bump_versions(self, changes: Dict[str, str], *, note: str = "") -> Dict[str, Any]:
        """Update version info."""
        if not isinstance(changes, dict) or not changes:
            raise ValueError("bump_versions: changes must be a non-empty dict.")

        data = self._read_json(self.versions_path)

        for k, v in changes.items():
            if k in ("prompt_version", "rubric_version", "retrieval_policy_version"):
                data[k] = str(v)

        entry = {
            "ts": self._now_ts(),
            "changes": {k: str(v) for k, v in changes.items()},
            "note": str(note)[:500],
        }

        changelog = data.get("changelog", [])
        if not isinstance(changelog, list):
            changelog = []
        changelog.append(entry)

        data["changelog"] = changelog
        data["updated_at"] = self._now_ts()

        self._write_json(self.versions_path, data)
        return data

    def get_versions(self) -> Dict[str, Any]:
        """Return current versions.json content."""
        return self._read_json(self.versions_path)

    # -------------------------------
    # File helpers
    # -------------------------------

    @staticmethod
    def _now_ts() -> int:
        return int(time.time())

    @staticmethod
    def _read_json(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {}
        return obj

    @staticmethod
    def _write_json(path: str, obj: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)