
# Purpose:
# - Centralize all shared data structures used across modules:
#     - RAG (ProblemSpec, ContextChunk)
#     - Answering (AnswerCandidate)
#     - Scoring (CandidateScore, SelectionResult)
#     - Student grading (StudentGradeResult)
#     - Common request/response payloads (optional, for API layer)
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional




AnswerStyle = Literal["brief", "detailed"]
CorrectnessLabel = Literal["correct", "incorrect", "partial"]

# RAG-related schemas
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

# Answering schemas
@dataclass
class AnswerCandidate:
    """
    One candidate answer produced by the model.
    """
    candidate_id: str             
    style: AnswerStyle
    final_answer: str
    steps: str
    citations: List[str]           
    raw_model_output: str          

# Scoring / selection schemas


@dataclass
class CandidateScore:
    """
    Scoring result for a single candidate.
    """
    candidate_id: str
    total: int                     
    rule_score: int                
    llm_score: int                 
    breakdown: Dict[str, int]     
    issues: List[str]


@dataclass
class SelectionResult:
    """
    Final selection output after scoring.
    """
    best_candidate_id: str
    best_answer: AnswerCandidate
    scores: List[CandidateScore]

# Student grading schemas

@dataclass
class StudentGradeResult:
    """
    Student grading output.
    """
    is_correct: CorrectnessLabel
    score: int                         
    feedback: str
    rubric_breakdown: Dict[str, int]   
    error_tags: List[str]

# Helpers
def to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert a dataclass instance to dict safely.
    If obj is not a dataclass, raises TypeError.
    """
    return asdict(obj)