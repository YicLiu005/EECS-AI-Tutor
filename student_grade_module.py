# student_grade_module.py
# Grade student answers from text or images.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from gemini_adapter import GeminiAdapter
from rag_module import RAGModule, ContextChunk


CorrectnessLabel = Literal["correct", "incorrect", "partial"]


@dataclass
class StudentGradeResult:
    is_correct: CorrectnessLabel
    score: int                 # 0-100
    feedback: str
    rubric_breakdown: Dict[str, int]
    error_tags: List[str]
    extra: Dict[str, Any]      # debug info

class StudentGradingModule:
    """Grade student answers."""

    def __init__(
        self,
        gemini: Optional[GeminiAdapter] = None,
        rag: Optional[RAGModule] = None,
        judge_model: Optional[str] = None,
        *,
        max_context_chars: int = 6000,
    ) -> None:
        self.gemini = gemini or GeminiAdapter()
        self.rag = rag or RAGModule()
        self.judge_model = judge_model
        self.max_context_chars = int(max_context_chars)

    # ---------------------------------------------------------
    # Unified entry
    # ---------------------------------------------------------

    def grade_student(
        self,
        problem_text: str,
        contexts: List[ContextChunk],
        *,
        student_answer_text: Optional[str] = None,
        student_answer_image_bytes: Optional[bytes] = None,
        student_answer_image_mime: str = "image/png",
        reference_answer: Optional[str] = None,
        subject: str = "",
        grade_level: str = "",
    ) -> StudentGradeResult:
        """Grade a student answer from text or image."""
        if not isinstance(problem_text, str) or not problem_text.strip():
            raise ValueError("grade_student: problem_text must be a non-empty string.")

        # Normalize student answer into text
        extracted_debug: Dict[str, Any] = {}
        normalized_student_answer = ""

        if student_answer_text and student_answer_text.strip():
            normalized_student_answer = student_answer_text.strip()
            extracted_debug = {
                "student_answer_source": "text",
                "extracted_student_answer": normalized_student_answer,
            }
        elif student_answer_image_bytes:
            parsed = self.rag.parse_student_answer_image(
                student_answer_image_bytes,
                mime_type=student_answer_image_mime,
            )
            student_text = str(parsed.get("student_answer_text", "")).strip()
            work = str(parsed.get("work", "")).strip()

            combined = student_text
            if work:
                combined = f"{student_text}\n\nWork:\n{work}" if student_text else f"Work:\n{work}"

            if not combined.strip():
                raise ValueError("grade_student: extracted student answer from image is empty.")

            normalized_student_answer = combined.strip()
            extracted_debug = {
                "student_answer_source": "image",
                "extracted_student_answer": normalized_student_answer,
                "vision_parse": parsed,
            }
        else:
            raise ValueError("grade_student: provide either student_answer_text or student_answer_image_bytes.")

        # Grade
        context_block = self._build_context_block(contexts)

        ref = reference_answer.strip() if isinstance(reference_answer, str) and reference_answer.strip() else None

        sys = self._judge_system_prompt(subject=subject, grade_level=grade_level)
        prompt = self._judge_prompt(
            problem_text=problem_text.strip(),
            context_block=context_block,
            student_answer=normalized_student_answer,
            reference_answer=ref,
        )

        raw = self.gemini.generate_text(prompt, system_prompt=sys, model=self.judge_model)
        result = self._parse_grade_json(raw)

        # Attach extra info
        result.extra = {
            **extracted_debug,
            "judge_raw_head": raw[:400],
        }
        return result

    # ---------------------------------------------------------
    # Judge prompting
    # ---------------------------------------------------------

    def _judge_system_prompt(self, *, subject: str, grade_level: str) -> str:
        subject_line = f"Subject: {subject}" if subject else "Subject: (not specified)"
        grade_line = f"Grade: {grade_level}" if grade_level else "Grade: (not specified)"
        return f"""
You are a strict elementary-school teacher grading a student's answer.
Use ONLY the QUESTION, the CONTEXT, and (if provided) the REFERENCE_ANSWER.
Do NOT invent a reference answer if it is not provided.
If reference answer is missing, judge based on logical consistency with question and context.
Return ONLY valid JSON. No markdown.
{subject_line}
{grade_line}
""".strip()

    def _judge_prompt(
        self,
        *,
        problem_text: str,
        context_block: str,
        student_answer: str,
        reference_answer: Optional[str],
    ) -> str:
        ref_part = f"REFERENCE_ANSWER:\n{reference_answer}\n" if reference_answer else "REFERENCE_ANSWER:\n(None)\n"

        return f"""
QUESTION:
{problem_text}

CONTEXT:
{context_block}

{ref_part}
STUDENT_ANSWER:
{student_answer}

Output ONLY valid JSON with:
{{
  "is_correct": "correct" | "incorrect" | "partial",
  "score": 0-100,
  "rubric_breakdown": {{
    "correctness": 0-100,
    "reasoning": 0-100,
    "clarity": 0-100,
    "completeness": 0-100
  }},
  "feedback": "short, kid-friendly feedback with what to improve",
  "error_tags": ["unit_error", "calculation_error", "missing_step", "misread_question", "..."]
}}

Grading rules:
- If REFERENCE_ANSWER is provided: grade against it.
- If REFERENCE_ANSWER is not provided: grade based on whether the student's answer is consistent with QUESTION and CONTEXT.
- Give partial credit when reasoning is mostly right but has minor mistakes.
- Feedback should be encouraging and clear for a child.
""".strip()

    # ---------------------------------------------------------
    # Parse judge JSON
    # ---------------------------------------------------------

    def _parse_grade_json(self, raw: str) -> StudentGradeResult:
        obj, err = self.gemini.try_parse_json(raw)
        if obj is None:
            raise ValueError(f"Judge output is not valid JSON. parse_error={err}. output_head={raw[:300]}")

        is_correct = str(obj.get("is_correct", "incorrect")).lower().strip()
        if is_correct not in ("correct", "incorrect", "partial"):
            is_correct = "incorrect"

        score = obj.get("score", 0)
        try:
            score = int(round(float(score)))
        except Exception:
            score = 0
        score = max(0, min(100, score))

        feedback = str(obj.get("feedback", "")).strip() or "No feedback provided."

        breakdown_raw = obj.get("rubric_breakdown", {})
        if not isinstance(breakdown_raw, dict):
            breakdown_raw = {}

        breakdown: Dict[str, int] = {}
        for k in ["correctness", "reasoning", "clarity", "completeness"]:
            v = breakdown_raw.get(k, 0)
            try:
                v_int = int(round(float(v)))
            except Exception:
                v_int = 0
            breakdown[k] = max(0, min(100, v_int))

        error_tags = obj.get("error_tags", [])
        if not isinstance(error_tags, list):
            error_tags = []
        error_tags = [str(x) for x in error_tags][:10]

        return StudentGradeResult(
            is_correct=is_correct,  # type: ignore
            score=score,
            feedback=feedback,
            rubric_breakdown=breakdown,
            error_tags=error_tags,
            extra={},  # caller fills
        )

    # ---------------------------------------------------------
    # Context formatting
    # ---------------------------------------------------------

    def _build_context_block(self, contexts: List[ContextChunk]) -> str:
        if not contexts:
            return "(No context)"

        blocks: List[str] = []
        total = 0
        for c in contexts:
            piece = f"[{c.chunk_id}]\n{c.content}\n"
            if total + len(piece) > self.max_context_chars:
                break
            blocks.append(piece)
            total += len(piece)

        return "\n".join(blocks).strip()