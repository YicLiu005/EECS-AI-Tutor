# rag_module.py
# Handle image parsing, storage, and retrieval.

from __future__ import annotations

import json
import os
import pickle
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from gemini_adapter import GeminiAdapter


@dataclass
class ProblemSpec:
    """Structured representation of a question extracted from an image or provided as text."""
    problem_text: str
    options: List[str]
    figure_desc: str
    constraints: List[str]
    grade_level: str = ""
    subject: str = ""


@dataclass
class ContextChunk:
    """A retrieved knowledge chunk."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float  # similarity score


class RAGModule:
    """Manage parsing, storage, and retrieval."""

    def __init__(
        self,
        gemini: Optional[GeminiAdapter] = None,
        storage_dir: str = "storage",
        kb_path: str = "kb.jsonl",
        vectordb_path: str = "vectordb.pkl",
        chunk_size_chars: int = 800,
        chunk_overlap_chars: int = 120,
        default_top_k: int = 5,
    ) -> None:
        self.gemini = gemini or GeminiAdapter()

        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.kb_file = os.path.join(self.storage_dir, kb_path)
        self.vectordb_file = os.path.join(self.storage_dir, vectordb_path)

        self.chunk_size_chars = int(chunk_size_chars)
        self.chunk_overlap_chars = int(chunk_overlap_chars)
        self.default_top_k = int(default_top_k)

        # Ensure vector DB exists
        if not os.path.exists(self.vectordb_file):
            self._save_db({"chunks": [], "embeddings": np.zeros((0, 0), dtype=np.float32)})

    # =========================================================
    # Image parsing
    # =========================================================

    def parse_problem_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
    ) -> ProblemSpec:
        """
        Parse a problem/question from an image WITHOUT saving into KB/vectordb.
        """
        raw_json_text = self.gemini.vision_to_text(image_bytes, mime_type=mime_type)

        obj, err = self.gemini.try_parse_json(raw_json_text)
        if obj is None:
            raise ValueError(f"Problem vision output is not valid JSON. parse_error={err}. output_head={raw_json_text[:300]}")

        return ProblemSpec(
            problem_text=str(obj.get("problem_text", "")).strip(),
            options=list(obj.get("options", [])) if isinstance(obj.get("options", []), list) else [],
            figure_desc=str(obj.get("figure_desc", "")).strip(),
            constraints=list(obj.get("constraints", [])) if isinstance(obj.get("constraints", []), list) else [],
            grade_level=str(obj.get("grade_level", "")).strip(),
            subject=str(obj.get("subject", "")).strip(),
        )

    def parse_student_answer_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
    ) -> Dict[str, Any]:
        """Extract student answer text from an image."""
        task_hint = """
You are given an image that contains a student's answer (handwriting or typed).
Extract the student's final answer and any shown work.

Return ONLY valid JSON with fields:
- student_answer_text: string (the final answer the student wrote, as text)
- work: string (optional; any steps/work shown; empty if none)
- confidence: number between 0 and 1 (best guess)
- notes: string (optional; e.g., unreadable parts)
""".strip()

        raw_json_text = self.gemini.vision_to_text(
            image_bytes,
            mime_type=mime_type,
            task_hint=task_hint,
        )

        obj, err = self.gemini.try_parse_json(raw_json_text)
        if obj is None:
            raise ValueError(f"Student answer vision output is not valid JSON. parse_error={err}. output_head={raw_json_text[:300]}")
        if not isinstance(obj, dict):
            raise ValueError("Student answer vision output JSON is not an object.")

        # Normalize required fields
        out: Dict[str, Any] = {
            "student_answer_text": str(obj.get("student_answer_text", "")).strip(),
            "work": str(obj.get("work", "")).strip(),
            "confidence": obj.get("confidence", 0.0),
            "notes": str(obj.get("notes", "")).strip(),
        }

        try:
            out["confidence"] = float(out["confidence"])
        except Exception:
            out["confidence"] = 0.0

        out["confidence"] = max(0.0, min(1.0, float(out["confidence"])))
        return out

    # =========================================================
    # KB ingest
    # =========================================================

    def ingest_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ProblemSpec, int]:
        """Save an image into the knowledge base."""
        doc_id = self._new_doc_id(prefix="img")

        problem = self.parse_problem_image(image_bytes, mime_type=mime_type)

        canonical_text = self._problem_to_canonical_text(problem)

        meta = metadata.copy() if metadata else {}
        meta.update({
            "doc_type": "image_question",
            "grade_level": problem.grade_level,
            "subject": problem.subject,
        })

        # Write raw record
        self._append_kb_raw(doc_id=doc_id, raw_type="image", raw_payload={
            "problem_text": problem.problem_text,
            "options": problem.options,
            "figure_desc": problem.figure_desc,
            "constraints": problem.constraints,
            "grade_level": problem.grade_level,
            "subject": problem.subject,
        }, metadata=meta)

        chunks = self._chunk_text(canonical_text)
        chunks_added = self._upsert_chunks(doc_id=doc_id, chunks=chunks, metadata=meta)

        return doc_id, problem, chunks_added

    def ingest_text(
        self,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int]:
        """
        Ingest plain text into the KB & vector DB.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("ingest_text: 'text' must be a non-empty string.")

        doc_id = self._new_doc_id(prefix="txt")
        meta = metadata.copy() if metadata else {}
        meta.update({"doc_type": "text"})

        self._append_kb_raw(doc_id=doc_id, raw_type="text", raw_payload={"text": text}, metadata=meta)

        chunks = self._chunk_text(text)
        chunks_added = self._upsert_chunks(doc_id=doc_id, chunks=chunks, metadata=meta)

        return doc_id, chunks_added

    # =========================================================
    # Retrieval
    # =========================================================

    def retrieve(
        self,
        query_text: str,
        *,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ContextChunk]:
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("retrieve: 'query_text' must be a non-empty string.")

        k = int(top_k) if top_k is not None else self.default_top_k
        if k <= 0:
            raise ValueError("retrieve: top_k must be positive.")

        db = self._load_db()
        chunks: List[Dict[str, Any]] = db["chunks"]
        emb: np.ndarray = db["embeddings"]

        if len(chunks) == 0:
            return []

        qvec = self.gemini.embed_text([query_text], task_type="RETRIEVAL_QUERY")[0]
        q = np.asarray(qvec, dtype=np.float32)

        if emb.shape[0] != len(chunks):
            raise RuntimeError("Vector DB corrupted: embeddings rows do not match chunks length.")
        if emb.shape[0] == 0:
            return []
        if emb.shape[1] != q.shape[0]:
            raise RuntimeError(
                f"Embedding dimension mismatch. db_dim={emb.shape[1]} query_dim={q.shape[0]}."
            )

        idxs = self._apply_filters(chunks, filters)
        if len(idxs) == 0:
            return []

        sub_emb = emb[idxs]
        scores = self._cosine_similarity_matrix(sub_emb, q)

        topk_local = min(k, scores.shape[0])
        best_local = np.argpartition(-scores, topk_local - 1)[:topk_local]
        best_sorted = best_local[np.argsort(-scores[best_local])]

        results: List[ContextChunk] = []
        for rank_i in best_sorted:
            global_i = idxs[int(rank_i)]
            ch = chunks[global_i]
            results.append(
                ContextChunk(
                    chunk_id=ch["chunk_id"],
                    doc_id=ch["doc_id"],
                    content=ch["content"],
                    metadata=ch.get("metadata", {}),
                    score=float(scores[int(rank_i)]),
                )
            )
        return results

    # =========================================================
    # Helpers
    # =========================================================

    def _chunk_text(self, text: str) -> List[str]:
        s = text.strip()
        if not s:
            return []
        size = self.chunk_size_chars
        overlap = self.chunk_overlap_chars
        if size <= 0:
            raise ValueError("chunk_size_chars must be > 0.")
        if overlap < 0:
            raise ValueError("chunk_overlap_chars must be >= 0.")
        if overlap >= size:
            overlap = max(0, size // 4)

        chunks: List[str] = []
        start = 0
        n = len(s)
        while start < n:
            end = min(n, start + size)
            chunk = s[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            start = max(0, end - overlap)
        return chunks

    def _problem_to_canonical_text(self, problem: ProblemSpec) -> str:
        parts = []
        if problem.subject:
            parts.append(f"[Subject] {problem.subject}")
        if problem.grade_level:
            parts.append(f"[Grade] {problem.grade_level}")
        if problem.problem_text:
            parts.append(f"[Problem] {problem.problem_text}")
        if problem.options:
            opts = "\n".join([f"- {o}" for o in problem.options])
            parts.append(f"[Options]\n{opts}")
        if problem.figure_desc:
            parts.append(f"[Figure]\n{problem.figure_desc}")
        if problem.constraints:
            cons = "\n".join([f"- {c}" for c in problem.constraints])
            parts.append(f"[Constraints]\n{cons}")
        return "\n\n".join(parts).strip()

    def _append_kb_raw(
        self,
        *,
        doc_id: str,
        raw_type: str,
        raw_payload: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        record = {
            "doc_id": doc_id,
            "raw_type": raw_type,
            "raw_payload": raw_payload,
            "metadata": metadata,
        }
        with open(self.kb_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_db(self) -> Dict[str, Any]:
        with open(self.vectordb_file, "rb") as f:
            db = pickle.load(f)

        if "chunks" not in db or "embeddings" not in db:
            raise RuntimeError("vectordb.pkl invalid: missing keys.")

        emb = db["embeddings"]
        if isinstance(emb, list):
            emb = np.asarray(emb, dtype=np.float32)
        if not isinstance(emb, np.ndarray):
            raise RuntimeError("vectordb.pkl invalid: embeddings not array-like.")

        db["embeddings"] = emb.astype(np.float32, copy=False)
        return db

    def _save_db(self, db: Dict[str, Any]) -> None:
        if "embeddings" in db and isinstance(db["embeddings"], list):
            db["embeddings"] = np.asarray(db["embeddings"], dtype=np.float32)
        with open(self.vectordb_file, "wb") as f:
            pickle.dump(db, f)

    def _upsert_chunks(
        self,
        *,
        doc_id: str,
        chunks: List[str],
        metadata: Dict[str, Any],
    ) -> int:
        if not chunks:
            return 0

        db = self._load_db()
        chunk_records: List[Dict[str, Any]] = db["chunks"]
        emb: np.ndarray = db["embeddings"]

        vecs = self.gemini.embed_text(chunks, task_type="RETRIEVAL_DOCUMENT")
        mat = np.asarray(vecs, dtype=np.float32)

        if emb.size == 0:
            emb = mat
        else:
            if emb.shape[1] != mat.shape[1]:
                raise RuntimeError(f"Embedding dimension mismatch: existing={emb.shape[1]} new={mat.shape[1]}.")
            emb = np.vstack([emb, mat])

        for content in chunks:
            chunk_records.append({
                "chunk_id": self._new_chunk_id(),
                "doc_id": doc_id,
                "content": content,
                "metadata": metadata,
            })

        db["chunks"] = chunk_records
        db["embeddings"] = emb
        self._save_db(db)
        return len(chunks)

    def _apply_filters(self, chunks: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> List[int]:
        if not filters:
            return list(range(len(chunks)))
        idxs: List[int] = []
        for i, ch in enumerate(chunks):
            meta = ch.get("metadata", {}) or {}
            ok = True
            for k, v in filters.items():
                if meta.get(k) != v:
                    ok = False
                    break
            if ok:
                idxs.append(i)
        return idxs

    @staticmethod
    def _cosine_similarity_matrix(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        mat_norm = np.linalg.norm(mat, axis=1) + 1e-12
        vec_norm = np.linalg.norm(vec) + 1e-12
        dot = mat @ vec
        return dot / (mat_norm * vec_norm)

    @staticmethod
    def _new_doc_id(prefix: str = "doc") -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    @staticmethod
    def _new_chunk_id() -> str:
        return f"chunk_{uuid.uuid4().hex}"