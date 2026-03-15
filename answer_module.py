# answer_module.py
# Generate answer candidates from the question and context.

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from gemini_adapter import GeminiAdapter


AnswerStyle = Literal["brief", "detailed"]


@dataclass
class ContextChunk:
    """Context used for answering."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


@dataclass
class AnswerCandidate:
    """One candidate answer produced by the model."""
    candidate_id: str
    style: AnswerStyle
    final_answer: str
    steps: str
    citations: List[str]          # list of chunk_id
    raw_model_output: str         # full model output for debugging/auditing


class AnswerModule:
    """Generate answer candidates."""

    def __init__(
        self,
        gemini: Optional[GeminiAdapter] = None,
        text_model: Optional[str] = None,
        max_context_chars: int = 6000,
    ) -> None:
        self.gemini = gemini or GeminiAdapter()
        self.text_model = text_model  # optional override
        self.max_context_chars = int(max_context_chars)

    def generate_candidates(
        self,
        problem_text: str,
        contexts: List[ContextChunk],
        *,
        style: AnswerStyle = "detailed",
        subject: str = "",
        grade_level: str = "",
    ) -> List[AnswerCandidate]:
        """Generate answer candidates."""

        if not isinstance(problem_text, str) or not problem_text.strip():
            raise ValueError("generate_candidates: problem_text must be a non-empty string.")

        # Build a compact context block for grounding
        context_block = self._build_context_block(contexts)

        # Candidate prompts (slightly different instructions to encourage diversity)
        prompts = self._build_three_prompts(
            problem_text=problem_text.strip(),
            context_block=context_block,
            style=style,
            subject=subject,
            grade_level=grade_level,
        )

        candidates: List[AnswerCandidate] = []
        for i, p in enumerate(prompts):
            raw = self.gemini.generate_text(
                p["prompt"],
                system_prompt=p["system_prompt"],
                model=self.text_model,
            )

            # Parse the model output into structured fields
            final_answer, steps, cites = self._parse_candidate_output(raw, contexts)

            # Enforce at least one citation if contexts exist
            if contexts and not cites:
                # If model failed to cite, fall back to citing top-1 context
                cites = [contexts[0].chunk_id]

            candidates.append(
                AnswerCandidate(
                    candidate_id=p["candidate_id"],
                    style=style,
                    final_answer=final_answer,
                    steps=steps,
                    citations=cites,
                    raw_model_output=raw,
                )
            )

        return candidates

    # -------------------------------
    # Prompt building
    # -------------------------------

    def _build_three_prompts(
        self,
        *,
        problem_text: str,
        context_block: str,
        style: AnswerStyle,
        subject: str,
        grade_level: str,
    ) -> List[Dict[str, str]]:
        """Build prompts for candidates."""

        sys = self._system_prompt(subject=subject, grade_level=grade_level)

        style_instructions = self._style_instructions(style)

        base = f"""
You are solving an elementary-school question.

{style_instructions}

Grounding rules:
- Use the provided CONTEXT to answer the question if it is helpful.
- If the context is empty or does not contain the answer, you MAY use your own general knowledge to help the student (especially for basic math or general facts).
- If you use information from the context, you MUST cite it like [CITE:chunk_id]. If you use your own knowledge, you don't need to cite.
- Output format must be exactly:
FINAL_ANSWER:
...
STEPS:
...
CITATIONS:
- chunk_id
- chunk_id
""".strip()

        prompt_a = f"""
{base}

QUESTION:
{problem_text}

CONTEXT:
{context_block}
""".strip()

        prompt_b = f"""
{base}

Extra requirement:
- Double-check units, hidden conditions, and common traps for kids.
- If there are options, verify which option matches.

QUESTION:
{problem_text}

CONTEXT:
{context_block}
""".strip()

        prompt_c = f"""
{base}

Extra requirement:
- First give an intuitive explanation (kid-friendly),
  then provide a short formal step-by-step solution.

QUESTION:
{problem_text}

CONTEXT:
{context_block}
""".strip()

        return [
            {"candidate_id": "A", "system_prompt": sys, "prompt": prompt_a}
        ]

    def _system_prompt(self, *, subject: str, grade_level: str) -> str:
        """
        System prompt: sets global behavior.
        """
        subject_line = f"Subject: {subject}" if subject else "Subject: (not specified)"
        grade_line = f"Grade: {grade_level}" if grade_level else "Grade: (not specified)"
        return f"""
You are a helpful tutor for elementary school students.
Be accurate, clear, and age-appropriate.
{subject_line}
{grade_line}
Do not include any unsafe content. Do not mention policy.
""".strip()

    def _style_instructions(self, style: AnswerStyle) -> str:
        """
        Style-specific guidance.
        """
        if style == "brief":
            return """
Answer style: BRIEF
- Keep it short.
- Provide the final answer and 1-3 bullet steps max.
- No long explanations.
""".strip()

        return """
Answer style: DETAILED
- Explain step-by-step.
- Keep language simple and kid-friendly.
- Show intermediate steps clearly.
""".strip()

    # -------------------------------
    # Context handling
    # -------------------------------

    def _build_context_block(self, contexts: List[ContextChunk]) -> str:
        """
        Build a context block with chunk IDs and content, truncated by max_context_chars.
        """
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

        if not blocks:
            return "(Context truncated)"

        return "\n".join(blocks).strip()

    # -------------------------------
    # Output parsing
    # -------------------------------

    def _parse_candidate_output(
        self,
        raw: str,
        contexts: List[ContextChunk],
    ) -> tuple[str, str, List[str]]:
        """Parse model output into answer parts."""
        text = (raw or "").strip()
        if not text:
            raise ValueError("Model output is empty.")

        final_answer = ""
        steps = ""
        citations: List[str] = []

        # Very lightweight section parsing (robust enough for MVP)
        def _get_section(label: str, next_labels: List[str]) -> str:
            start = text.find(label)
            if start < 0:
                return ""
            start += len(label)
            # Find nearest next label
            end_positions = [text.find(nl, start) for nl in next_labels]
            end_positions = [p for p in end_positions if p >= 0]
            end = min(end_positions) if end_positions else len(text)
            return text[start:end].strip()

        final_answer = _get_section("FINAL_ANSWER:", ["STEPS:", "CITATIONS:"])
        steps = _get_section("STEPS:", ["CITATIONS:"])
        cites_block = _get_section("CITATIONS:", [])

        # Parse citations list lines "- chunk_id"
        for line in cites_block.splitlines():
            line = line.strip()
            if line.startswith("-"):
                cid = line[1:].strip()
                if cid:
                    citations.append(cid)

        # Also parse inline citations [CITE:chunk_id]
        inline = self._extract_inline_citations(text)
        for cid in inline:
            if cid not in citations:
                citations.append(cid)

        # Keep only citations that exist in provided contexts (if any)
        if contexts:
            valid_ids = {c.chunk_id for c in contexts}
            citations = [cid for cid in citations if cid in valid_ids]

        # Fallback if sections missing: treat whole text as final answer
        if not final_answer:
            final_answer = text

        if not steps:
            steps = "(No steps provided)"

        return final_answer.strip(), steps.strip(), citations

    @staticmethod
    def _extract_inline_citations(text: str) -> List[str]:
        """
        Extract citations like [CITE:chunk_xxx] from text.
        """
        cites: List[str] = []
        marker = "[CITE:"
        i = 0
        while True:
            j = text.find(marker, i)
            if j < 0:
                break
            k = text.find("]", j)
            if k < 0:
                break
            cid = text[j + len(marker):k].strip()
            if cid:
                cites.append(cid)
            i = k + 1
        return cites