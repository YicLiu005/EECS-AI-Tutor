# main.py
# FastAPI app for the project.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI, File, UploadFile, Form

from rag_module import RAGModule
from answer_module import AnswerModule
from answer_score_module import AnswerScoringModule
from student_grade_module import StudentGradingModule
from learning_module import LearningModule


AnswerStyle = Literal["brief", "detailed"]

app = FastAPI(title="KidLearnAI", version="0.1.0")

# Shared module instances
rag = RAGModule(storage_dir="storage")
answerer = AnswerModule()
scorer = AnswerScoringModule(use_llm_judge=True)
student_grader = StudentGradingModule(rag=rag)   # <<< CHANGED (shared rag)
learner = LearningModule(storage_dir="storage")


# -------------------------------
# Helpers
# -------------------------------

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
    """Get problem text from text or image."""
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


def _extract_student_answer(
    *,
    student_answer_text: str,
    student_answer_file: Optional[UploadFile],
) -> Dict[str, Any]:
    """Get student answer from text or image."""
    if student_answer_text and student_answer_text.strip():
        return {
            "student_answer_text": student_answer_text.strip(),
            "student_answer_image_bytes": None,
            "student_answer_mime": None,
        }

    if student_answer_file is None:
        raise ValueError("Student answer missing: provide student_answer_text or upload student_answer_file.")

    image_bytes = student_answer_file.file.read()
    mime_type = student_answer_file.content_type or "image/png"

    return {
        "student_answer_text": None,
        "student_answer_image_bytes": image_bytes,
        "student_answer_mime": mime_type,
    }


# -------------------------------
# Ingest endpoint
# -------------------------------

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


# -------------------------------
# Endpoint: ask (problem as text OR image)
# -------------------------------

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

    selection = scorer.select_best(
        problem_text=question,
        contexts=contexts,
        candidates=candidates,
        style=style,
        subject=subject,
        grade_level=grade_level,
    )

    best = selection.best_answer
    best_score_total = next(s.total for s in selection.scores if s.candidate_id == selection.best_candidate_id)

    learner.log_interaction({
        "type": "ask",
        "style": style,
        "subject": subject,
        "grade_level": grade_level,
        "problem_input": {"problem_text_used": bool(problem_text.strip()), "problem_image_used": bool(problem_file is not None and not problem_text.strip())},
        "problem_spec": prob["problem_spec"],
        "question": question,
        "contexts": _compact_contexts(contexts),
        "candidates": [
            {"candidate_id": c.candidate_id, "citations": c.citations, "answer": c.raw_model_output}
            for c in candidates
        ],
        "scores": [
            {"candidate_id": s.candidate_id, "total": s.total, "breakdown": s.breakdown, "issues": s.issues}
            for s in selection.scores
        ],
        "best_candidate_id": selection.best_candidate_id,
    })

    learner.maybe_writeback_to_kb(
        kb_ingest_text_fn=rag.ingest_text,
        problem_text=question,
        chosen_answer_text=best.raw_model_output,
        chosen_score_total=best_score_total,
        metadata={"subject": subject, "grade_level": grade_level},
    )

    return {
        "best_candidate_id": selection.best_candidate_id,
        "best_answer": best.raw_model_output,
        "citations": best.citations,
        "scores": [
            {"candidate_id": s.candidate_id, "total": s.total, "breakdown": s.breakdown, "issues": s.issues}
            for s in selection.scores
        ],
        "candidates": [
            {"candidate_id": c.candidate_id, "answer": c.raw_model_output, "citations": c.citations}
            for c in candidates
        ],
        "problem_spec": prob["problem_spec"],
    }


# -------------------------------
# Ask endpoint
# -------------------------------

@app.post("/grade")
async def grade(
    top_k: int = Form(5),
    subject: str = Form(""),
    grade_level: str = Form(""),
    reference_answer: str = Form(""),
    problem_text: str = Form(""),
    problem_file: UploadFile = File(None),
    student_answer_text: str = Form(""),
    student_answer_file: UploadFile = File(None),
):
    prob = _extract_problem_text(problem_text=problem_text, problem_file=problem_file)
    question = prob["problem_text"]

    ans = _extract_student_answer(student_answer_text=student_answer_text, student_answer_file=student_answer_file)

    contexts = rag.retrieve(
        question,
        top_k=top_k,
        filters={"subject": subject} if subject else None,
    )

    ref = reference_answer.strip() if reference_answer and reference_answer.strip() else None

    result = student_grader.grade_student(
        problem_text=question,
        contexts=contexts,
        student_answer_text=ans["student_answer_text"],
        student_answer_image_bytes=ans["student_answer_image_bytes"],
        student_answer_image_mime=(ans["student_answer_mime"] or "image/png"),
        reference_answer=ref,
        subject=subject,
        grade_level=grade_level,
    )

    learner.log_interaction({
        "type": "grade_student",
        "subject": subject,
        "grade_level": grade_level,
        "problem_input": {"problem_text_used": bool(problem_text.strip()), "problem_image_used": bool(problem_file is not None and not problem_text.strip())},
        "problem_spec": prob["problem_spec"],
        "question": question,
        "student_answer_input": {
            "student_text_used": bool(student_answer_text.strip()),
            "student_image_used": bool(student_answer_file is not None and not student_answer_text.strip()),
        },
        "contexts": _compact_contexts(contexts),
        "reference_answer": ref or "",
        "grade_result": {
            "is_correct": result.is_correct,
            "score": result.score,
            "feedback": result.feedback,
            "rubric_breakdown": result.rubric_breakdown,
            "error_tags": result.error_tags,
            "extra": result.extra,
        },
    })

    learner.update_error_patterns(result.error_tags)

    learner.maybe_add_to_evalset(
        problem_text=question,
        reference_answer=ref,
        chosen_answer=result.extra.get("extracted_student_answer", "") or (student_answer_text.strip() if student_answer_text else ""),
        metadata={"subject": subject, "grade_level": grade_level, "type": "student_submission"},
    )

    return {
        "problem_spec": prob["problem_spec"],
        "question": question,
        "is_correct": result.is_correct,
        "score": result.score,
        "feedback": result.feedback,
        "rubric_breakdown": result.rubric_breakdown,
        "error_tags": result.error_tags,
        "extracted_student_answer": result.extra.get("extracted_student_answer", ""),
        "debug": {
            "student_answer_source": result.extra.get("student_answer_source", ""),
        },
    }


# -------------------------------
# Grade endpoint
# -------------------------------

@app.post("/feedback")
async def feedback(
    question: str = Form(...),
    corrected_reference_answer: str = Form(...),
    note: str = Form(""),
    subject: str = Form(""),
    grade_level: str = Form(""),
):
    learner.log_interaction({
        "type": "feedback",
        "question": question,
        "corrected_reference_answer": corrected_reference_answer,
        "note": note,
        "subject": subject,
        "grade_level": grade_level,
    })

    learner.maybe_add_to_evalset(
        problem_text=question,
        reference_answer=corrected_reference_answer,
        chosen_answer="",
        metadata={"subject": subject, "grade_level": grade_level, "type": "human_correction", "note": note},
    )

    return {"ok": True}