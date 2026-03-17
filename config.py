
from __future__ import annotations

from typing import Final

# Gemini API
GEMINI_API_KEY: Final[str] = "AIzaSyA_IOnfx2P9-QQ3shV-8V9mdHQ1mBCGPrE"
GEMINI_API_BASE: Final[str] = "https://generativelanguage.googleapis.com/v1beta"

# Models
GEMINI_TEXT_MODEL: Final[str] = "gemini-2.5-flash"
GEMINI_VISION_MODEL: Final[str] = "gemini-2.5-flash"
GEMINI_EMBED_MODEL: Final[str] = "gemini-embedding-001"

# Default generation config
GEMINI_TEMPERATURE: Final[float] = 0.7
GEMINI_TOP_P: Final[float] = 0.95
GEMINI_TOP_K: Final[int] = 40
GEMINI_MAX_OUTPUT_TOKENS: Final[int] = 4096

# Request timeout (seconds)
GEMINI_TIMEOUT: Final[int] = 60

GEMINI_BASE_URL: Final[str] = GEMINI_API_BASE
GENERATION_MODEL: Final[str] = GEMINI_TEXT_MODEL
VISION_MODEL: Final[str] = GEMINI_VISION_MODEL
EMBEDDING_MODEL: Final[str] = GEMINI_EMBED_MODEL
# Storage (local files under STORAGE_DIR)


STORAGE_DIR: Final[str] = "storage"
KB_JSONL: Final[str] = "kb.jsonl"
VECTORDB_PKL: Final[str] = "vectordb.pkl"
LOGS_JSONL: Final[str] = "logs.jsonl"
ERROR_PATTERNS_JSON: Final[str] = "error_patterns.json"
EVALSET_JSONL: Final[str] = "evalset.jsonl"
VERSIONS_JSON: Final[str] = "versions.json"

# RAG settings
RAG_CHUNK_SIZE_CHARS: Final[int] = 800
RAG_CHUNK_OVERLAP_CHARS: Final[int] = 120
RAG_DEFAULT_TOP_K: Final[int] = 5
RAG_MAX_CONTEXT_CHARS: Final[int] = 6000

# Answer generation

ANSWER_NUM_CANDIDATES: Final[int] = 3
ANSWER_MAX_CONTEXT_CHARS: Final[int] = 6000
ANSWER_STYLE_BRIEF: Final[str] = "brief"
ANSWER_STYLE_DETAILED: Final[str] = "detailed"



SCORER_USE_LLM_JUDGE: Final[bool] = True
SCORER_RULE_WEIGHT: Final[float] = 0.35
SCORER_LLM_WEIGHT: Final[float] = 0.65
SCORER_CORRECTNESS_GATE: Final[int] = 60
STUDENT_GRADING_USE_LLM_JUDGE: Final[bool] = True
STUDENT_GRADING_MAX_CONTEXT_CHARS: Final[int] = 6000

# Learning / self-improvement


LEARNING_MIN_ANSWER_QUALITY_SCORE: Final[int] = 85
LEARNING_MAX_KB_WRITEBACK_CHARS: Final[int] = 4000
LEARNING_ALWAYS_ADD_TO_EVALSET: Final[bool] = False
LEARNING_ADD_TO_EVALSET_REQUIRES_REFERENCE: Final[bool] = True

# App versioning

APP_VERSION: Final[str] = "0.1.0"
PROMPT_VERSION: Final[str] = "v1"
RUBRIC_VERSION: Final[str] = "v1"
RETRIEVAL_POLICY_VERSION: Final[str] = "v1"