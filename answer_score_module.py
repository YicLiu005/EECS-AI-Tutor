# answer_score_module.py
# Score answer candidates and choose the best one.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from gemini_adapter import GeminiAdapter


AnswerStyle = Literal["brief", "detailed"]


@dataclass
class ContextChunk:
    """Context used for scoring."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


@dataclass
class AnswerCandidate:
    """Answer candidate used for scoring."""
    candidate_id: str
    style: AnswerStyle
    final_answer: str
    steps: str
    citations: List[str]
    raw_model_output: str


@dataclass
class CandidateScore:
    """Scoring result for a single candidate."""
    candidate_id: str
    total: int                       # 0-100
    rule_score: int                  # 0-100
    llm_score: int                   # 0-100
    breakdown: Dict[str, int]        # per-dimension breakdown
    issues: List[str]                # short reasons for deductions


@dataclass
class SelectionResult:
    """Final selection output."""
    best_candidate_id: str
    best_answer: AnswerCandidate
    scores: List[CandidateScore]


class AnswerScoringModule:
    """Score answers and select the best one."""

    def __init__(
        self,
        gemini: Optional[GeminiAdapter] = None,
        judge_model: Optional[str] = None,
        *,
        max_context_chars: int = 6000,
        use_llm_judge: bool = True,
        rule_weight: float = 0.35,
        llm_weight: float = 0.65,
    ) -> None:
        self.gemini = gemini or GeminiAdapter()
        self.judge_model = judge_model
        self.max_context_chars = int(max_context_chars)

        self.use_llm_judge = bool(use_llm_judge)
        self.rule_weight = float(rule_weight)
        self.llm_weight = float(llm_weight)

        if self.rule_weight < 0 or self.llm_weight < 0:
            raise ValueError("Weights must be non-negative.")
        if self.rule_weight == 0 and self.llm_weight == 0:
            raise ValueError("At least one of rule_weight or llm_weight must be > 0.")

    def select_best(
        self,
        problem_text: str,
        contexts: List[ContextChunk],
        candidates: List[AnswerCandidate],
        *,
        style: Optional[AnswerStyle] = None,
        subject: str = "",
        grade_level: str = "",
    ) -> SelectionResult:
        """
        Score each candidate and select the best one.

        Returns:
            SelectionResult with best candidate and all scores.
        """
        if not isinstance(problem_text, str) or not problem_text.strip():
            raise ValueError("select_best: problem_text must be a non-empty string.")
        if not candidates or len(candidates) < 1:
            raise ValueError("select_best: candidates must be a non-empty list.")

        # If style is provided, use it; otherwise infer from first candidate
        resolved_style: AnswerStyle = style or candidates[0].style

        # Build a compact context block for the LLM judge
        context_block = self._build_context_block(contexts)

        scores: List[CandidateScore] = []
        for cand in candidates:
            rule_score, rule_breakdown, rule_issues = self._rule_based_score(
                problem_text=problem_text,
                contexts=contexts,
                candidate=cand,
                style=resolved_style,
            )

            llm_score = 0
            llm_breakdown: Dict[str, int] = {}
            llm_issues: List[str] = []

            if self.use_llm_judge:
                llm_score, llm_breakdown, llm_issues = self._llm_judge_score(
                    problem_text=problem_text,
                    context_block=context_block,
                    candidate=cand,
                    style=resolved_style,
                    subject=subject,
                    grade_level=grade_level,
                )

            total = self._combine_scores(rule_score, llm_score)
            merged_breakdown = self._merge_breakdowns(rule_breakdown, llm_breakdown)
            merged_issues = rule_issues + llm_issues

            scores.append(
                CandidateScore(
                    candidate_id=cand.candidate_id,
                    total=total,
                    rule_score=rule_score,
                    llm_score=llm_score,
                    breakdown=merged_breakdown,
                    issues=merged_issues,
                )
            )

        # Selection rule:
        # 1) Hard gate: if LLM judge used, prefer candidates with correctness >= 60 if available
        # 2) Otherwise pick highest total; tie-break: higher correctness, then more citations
        best = self._pick_best(scores, candidates)

        best_candidate = next(c for c in candidates if c.candidate_id == best.candidate_id)
        return SelectionResult(
            best_candidate_id=best.candidate_id,
            best_answer=best_candidate,
            scores=scores,
        )

    # -------------------------------
    # Score combination / selection
    # -------------------------------

    def _combine_scores(self, rule_score: int, llm_score: int) -> int:
        rw = self.rule_weight
        lw = self.llm_weight
        denom = rw + lw
        if denom <= 0:
            return int(rule_score)
        val = (rw * rule_score + lw * llm_score) / denom
        return int(round(max(0.0, min(100.0, val))))

    @staticmethod
    def _merge_breakdowns(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        out = dict(a)
        for k, v in b.items():
            if k in out:
                # Average if both provide the same dimension
                out[k] = int(round((out[k] + v) / 2))
            else:
                out[k] = v
        return out

    def _pick_best(self, scores: List[CandidateScore], candidates: List[AnswerCandidate]) -> CandidateScore:
        # Build helper map for tie-breakers
        cand_map = {c.candidate_id: c for c in candidates}

        # Primary: total score
        # Tie-break: correctness (if present), then citations count
        def key_fn(s: CandidateScore) -> Tuple[int, int, int]:
            correctness = s.breakdown.get("correctness", 0)
            cites = len(cand_map[s.candidate_id].citations)
            return (s.total, correctness, cites)

        # Optional gate: if correctness exists, prefer those above threshold
        threshold = 60
        gated = [s for s in scores if s.breakdown.get("correctness", 100) >= threshold]
        pool = gated if gated else scores

        best = max(pool, key=key_fn)
        return best

    # -------------------------------
    # Rule-based scoring
    # -------------------------------

    def _rule_based_score(
        self,
        *,
        problem_text: str,
        contexts: List[ContextChunk],
        candidate: AnswerCandidate,
        style: AnswerStyle,
    ) -> Tuple[int, Dict[str, int], List[str]]:
        """
        Deterministic scoring signals; returns (score, breakdown, issues).
        """
        issues: List[str] = []
        breakdown: Dict[str, int] = {}

        # Start from 100 and deduct
        score = 100

        fa = (candidate.final_answer or "").strip()
        st = (candidate.steps or "").strip()

        # Basic presence checks
        if not fa:
            score -= 35
            issues.append("Missing final answer.")
        if style == "detailed":
            if not st or st == "(No steps provided)":
                score -= 20
                issues.append("Detailed style requires clear steps.")
        else:
            # brief style: steps can be short, but still should exist
            if not st or st == "(No steps provided)":
                score -= 10
                issues.append("Brief style should still include minimal steps.")

        # Citation requirement when context exists
        if contexts:
            if not candidate.citations:
                score -= 25
                issues.append("No citations provided despite having contexts.")
            else:
                # Validate citation IDs belong to contexts
                valid = {c.chunk_id for c in contexts}
                bad = [cid for cid in candidate.citations if cid not in valid]
                if bad:
                    score -= 10
                    issues.append("Contains invalid citations (not in retrieved contexts).")

        # Penalize excessive "Not found in context"
        nf = candidate.raw_model_output.lower().count("not found in context")
        if nf >= 3:
            score -= 10
            issues.append("Overuses 'Not found in context'.")

        # Length heuristics (simple)
        length = len(candidate.raw_model_output)
        if style == "brief":
            if length > 1600:
                score -= 10
                issues.append("Brief answer is too long.")
            if length < 120:
                score -= 8
                issues.append("Brief answer may be too short/insufficient.")
        else:
            if length < 250:
                score -= 10
                issues.append("Detailed answer seems too short.")
            if length > 3500:
                score -= 8
                issues.append("Detailed answer is excessively long.")

        score = max(0, min(100, score))

        # Provide a coarse breakdown
        breakdown["format"] = max(0, min(100, 100 - (10 if issues else 0)))
        breakdown["grounding"] = 100 if (not contexts or candidate.citations) else 70
        breakdown["completeness"] = 90 if (fa and st) else 60

        return int(score), breakdown, issues

    # -------------------------------
    # LLM judge scoring
    # -------------------------------

    def _llm_judge_score(
        self,
        *,
        problem_text: str,
        context_block: str,
        candidate: AnswerCandidate,
        style: AnswerStyle,
        subject: str,
        grade_level: str,
    ) -> Tuple[int, Dict[str, int], List[str]]:
        """
        Use Gemini as a judge to score a candidate more semantically.
        Returns (llm_score, breakdown, issues).
        """
        sys = self._judge_system_prompt(subject=subject, grade_level=grade_level)

        prompt = self._judge_prompt(
            problem_text=problem_text,
            context_block=context_block,
            candidate=candidate,
            style=style,
        )

        raw = self.gemini.generate_text(
            prompt,
            system_prompt=sys,
            model=self.judge_model,
        )

        # Expect JSON output; if parsing fails, degrade gracefully but still raise for now.
        obj, err = self.gemini.try_parse_json(raw)
        if obj is None:
            raise ValueError(f"Judge output is not valid JSON. parse_error={err}. output_head={raw[:300]}")

        breakdown = obj.get("breakdown", {})
        if not isinstance(breakdown, dict):
            breakdown = {}

        # Normalize breakdown values into ints 0-100
        norm_breakdown: Dict[str, int] = {}
        for k in ["correctness", "clarity", "grounding", "style_match", "completeness"]:
            v = breakdown.get(k, 0)
            try:
                v_int = int(round(float(v)))
            except Exception:
                v_int = 0
            norm_breakdown[k] = max(0, min(100, v_int))

        # Judge total score
        total = obj.get("total", None)
        if total is None:
            # Weighted average if total missing
            total = int(round(
                0.35 * norm_breakdown["correctness"]
                + 0.2 * norm_breakdown["grounding"]
                + 0.2 * norm_breakdown["clarity"]
                + 0.15 * norm_breakdown["completeness"]
                + 0.1 * norm_breakdown["style_match"]
            ))
        else:
            try:
                total = int(round(float(total)))
            except Exception:
                total = 0

        total = max(0, min(100, total))

        issues = obj.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        issues = [str(x) for x in issues][:8]

        return total, norm_breakdown, issues

    def _judge_system_prompt(self, *, subject: str, grade_level: str) -> str:
        subject_line = f"Subject: {subject}" if subject else "Subject: (not specified)"
        grade_line = f"Grade: {grade_level}" if grade_level else "Grade: (not specified)"
        return f"""
You are a strict but fair judge for an elementary-school tutoring assistant.
Evaluate the candidate answer using ONLY the given QUESTION and CONTEXT.
Do not assume facts not present in context.
{subject_line}
{grade_line}
Return ONLY valid JSON. Do not include markdown.
""".strip()

    def _judge_prompt(
        self,
        *,
        problem_text: str,
        context_block: str,
        candidate: AnswerCandidate,
        style: AnswerStyle,
    ) -> str:
        return f"""
You will score one candidate answer.

Style requirement:
- brief: short, direct, minimal steps
- detailed: step-by-step, kid-friendly explanation

QUESTION:
{problem_text}

CONTEXT:
{context_block}

CANDIDATE (raw):
{candidate.raw_model_output}

Now output ONLY valid JSON with:
{{
  "total": 0-100,
  "breakdown": {{
    "correctness": 0-100,
    "grounding": 0-100,
    "clarity": 0-100,
    "completeness": 0-100,
    "style_match": 0-100
  }},
  "issues": ["short reason 1", "short reason 2"]
}}

Scoring guidance:
- correctness: is the final answer correct and consistent with question/constraints?
- grounding: does it avoid hallucinating beyond context? Are citations consistent with context?
- clarity: understandable for kids
- completeness: key steps included (especially for detailed)
- style_match: matches requested style ({style})
""".strip()

    # -------------------------------
    # Context formatting
    # -------------------------------

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