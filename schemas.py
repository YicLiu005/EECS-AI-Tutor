# schemas.py
# Shared data types used across the project.

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional


# -------------------------------
# Common types
# -------------------------------

AnswerStyle = Literal["brief", "detailed"]
CorrectnessLabel = Literal["correct", "incorrect", "partial"]


# -------------------------------
# RAG-related schemas
# -------------------------------

@dataclass
class ProblemSpec:
    """
    Structured representation of a question extracted from an image or provided as text.
    """
    problem_text: str
    options: List[str]
    figure_desc: str
    constraints: List[str]
    grade_level: str = ""
    subject: str = ""


@dataclass
class ContextChunk:
    """
    A retrieved knowledge chunk.
    """
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


# -------------------------------
# Answering schemas
# -------------------------------

@dataclass
class AnswerCandidate:
    """Generated answer."""

    candidate_id: str              # "A" | "B" | "C"
    style: AnswerStyle
    final_answer: str
    steps: str
    citations: List[str]           # list of chunk_id
    raw_model_output: str          # full output (audit/debug)


# -------------------------------
# Scoring / selection schemas
# -------------------------------

@dataclass
class CandidateScore:
    """
    Scoring result for a single candidate.
    """
    candidate_id: str
    total: int                     # 0-100
    rule_score: int                # 0-100
    llm_score: int                 # 0-100
    breakdown: Dict[str, int]      # e.g. correctness/grounding/clarity...
    issues: List[str]


@dataclass
class SelectionResult:
    """
    Final selection output after scoring.
    """
    best_candidate_id: str
    best_answer: AnswerCandidate
    scores: List[CandidateScore]


# -------------------------------
# Student grading schemas
# -------------------------------

@dataclass
class StudentGradeResult:
    """
    Student grading output.
    """
    is_correct: CorrectnessLabel
    score: int                         # 0-100
    feedback: str
    rubric_breakdown: Dict[str, int]   # correctness/reasoning/clarity/completeness
    error_tags: List[str]


# -------------------------------
# Helpers
# -------------------------------

def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass to a dict."""
    return asdict(obj)