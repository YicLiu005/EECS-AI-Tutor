from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI, File, UploadFile, Form

from rag_module import RAGModule
from answer_module import AnswerModule


AnswerStyle = Literal["brief", "detailed"]

app = FastAPI(title="KidLearnAI", version="0.1.0")

# Shared module instances
rag = RAGModule(storage_dir="storage")
answerer = AnswerModule()

def _compact_contexts(contexts, max_chars: int = 200) -> List[Dict[str, Any]]:
    out = []
    for c in contexts:
        out.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "score": c.score,
            "content_head": (c.content[:max_chars] + "...") if len(c.content) > max_chars else c.content,
            "metadata": c.metadata,
        })
    return out


def _extract_problem_text(
    *,
    problem_text: str,
    problem_file: Optional[UploadFile],
) -> Dict[str, Any]:
    """
    Normalize problem input:
    - If problem_text is provided, use it
    - Else if problem_file is provided, parse ONLY (no KB write):
        rag.parse_problem_image(...) -> ProblemSpec
    Returns:
      { "problem_text": str, "problem_spec": dict|None }
    """
    if problem_text and problem_text.strip():
        return {"problem_text": problem_text.strip(), "problem_spec": None}

    if problem_file is None:
        raise ValueError("Problem input missing: provide problem_text or upload problem_file.")

    image_bytes = problem_file.file.read()
    mime_type = problem_file.content_type or "image/png"

    problem_spec = rag.parse_problem_image(image_bytes, mime_type=mime_type)

    if not problem_spec.problem_text.strip():
        raise ValueError("Failed to extract problem_text from problem image.")

    return {
        "problem_text": problem_spec.problem_text.strip(),
        "problem_spec": {
            "problem_text": problem_spec.problem_text,
            "options": problem_spec.options,
            "figure_desc": problem_spec.figure_desc,
            "constraints": problem_spec.constraints,
            "grade_level": problem_spec.grade_level,
            "subject": problem_spec.subject,
        },
    }

@app.post("/ingest")
async def ingest(
    mode: str = Form(..., description="image|text"),
    text: str = Form("", description="Text to ingest when mode=text"),
    file: UploadFile = File(None),
    subject: str = Form(""),
    grade_level: str = Form(""),
):
    meta = {"subject": subject, "grade_level": grade_level}

    if mode == "text":
        doc_id, chunks_added = rag.ingest_text(text, metadata=meta)
        return {"doc_id": doc_id, "chunks_added": chunks_added}

    if mode == "image":
        if file is None:
            raise ValueError("mode=image requires a file upload.")
        image_bytes = await file.read()
        mime_type = file.content_type or "image/png"
        doc_id, problem, chunks_added = rag.ingest_image(image_bytes, mime_type=mime_type, metadata=meta)
        return {
            "doc_id": doc_id,
            "chunks_added": chunks_added,
            "problem_spec": {
                "problem_text": problem.problem_text,
                "options": problem.options,
                "figure_desc": problem.figure_desc,
                "constraints": problem.constraints,
                "grade_level": problem.grade_level,
                "subject": problem.subject,
            },
        }

    raise ValueError("mode must be 'image' or 'text'.")

@app.post("/ask")
async def ask(
    style: AnswerStyle = Form("detailed"),
    top_k: int = Form(5),
    subject: str = Form(""),
    grade_level: str = Form(""),
    problem_text: str = Form(""),
    problem_file: UploadFile = File(None),
):
    prob = _extract_problem_text(problem_text=problem_text, problem_file=problem_file)
    question = prob["problem_text"]

    contexts = rag.retrieve(
        question,
        top_k=top_k,
        filters={"subject": subject} if subject else None,
    )

    candidates = answerer.generate_candidates(
        problem_text=question,
        contexts=contexts,
        style=style,
        subject=subject,
        grade_level=grade_level,
    )

    if not candidates:
        raise ValueError("No answer candidates were generated.")

    best = candidates[0]

    return {
        "best_candidate_id": best.candidate_id,
        "best_answer": best.raw_model_output,
        "citations": best.citations,
        "candidates": [
            {"candidate_id": c.candidate_id, "answer": c.raw_model_output, "citations": c.citations}
            for c in candidates
        ],
        "problem_spec": prob["problem_spec"],
        "contexts": _compact_contexts(contexts),
    }