# answer_module.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from gemini_adapter import GeminiAdapter

AnswerStyle = Literal["brief", "detailed"]


@dataclass
class ContextChunk:
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


@dataclass
class AnswerCandidate:
    candidate_id: str
    style: AnswerStyle
    final_answer: str
    steps: str
    citations: List[str]
    raw_model_output: str


class AnswerModule:
    """
    Single-answer generator with:
    - optional multi-turn chat history
    - optional textbook markdown retrieval:
        * textbook_md_path: one .md file
        * textbook_md_dir : a directory containing many .md files

    Citations policy:
    - CITATIONS section MUST be a bullet list with "- chunk_id" OR "- NONE"
    - No natural language allowed in CITATIONS
    - In FINAL_ANSWER and STEPS, grounded sentences/steps should be wrapped as:
      <cite chunk_id="..."> ... </cite>
    """

    def __init__(
        self,
        gemini: Optional[GeminiAdapter] = None,
        text_model: Optional[str] = None,
        max_context_chars: int = 20000,
        history_max_chars: int = 6000,
        textbook_md_path: Optional[str] = None,
        textbook_md_dir: Optional[str] = None,
        textbook_top_k: int = 5,
        textbook_min_score: float = 0.05,
        textbook_max_chars: int = 6000,
        vocab_boost: float = 1.5,
        reserve_vocab_slot: bool = True,
    ) -> None:
        self.gemini = gemini or GeminiAdapter()
        self.text_model = text_model
        self.max_context_chars = int(max_context_chars)
        self.history_max_chars = int(history_max_chars)

        self.textbook_md_path = textbook_md_path
        self.textbook_md_dir = textbook_md_dir
        self.textbook_top_k = int(textbook_top_k)
        self.textbook_min_score = float(textbook_min_score)
        self.textbook_max_chars = int(textbook_max_chars)

        self.vocab_boost = float(vocab_boost)
        self.reserve_vocab_slot = bool(reserve_vocab_slot)

        self._textbook_chunks: List[ContextChunk] = []
        if self.textbook_md_path:
            self._textbook_chunks.extend(self._load_textbook_md_path(self.textbook_md_path))
        if self.textbook_md_dir:
            self._textbook_chunks.extend(self._load_textbook_md_dir(self.textbook_md_dir))

    
    # Public API

    def generate_candidates(
        self,
        problem_text: str,
        contexts: List[ContextChunk],
        *,
        style: AnswerStyle = "detailed",
        subject: str = "",
        grade_level: str = "",
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> List[AnswerCandidate]:
        one = self.generate_answer(
            problem_text=problem_text,
            contexts=contexts,
            style=style,
            subject=subject,
            grade_level=grade_level,
            history=history,
        )
        return [one]

    def generate_answer(
        self,
        *,
        problem_text: str,
        contexts: List[ContextChunk],
        style: AnswerStyle = "detailed",
        subject: str = "",
        grade_level: str = "",
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> AnswerCandidate:
        if not isinstance(problem_text, str) or not problem_text.strip():
            raise ValueError("generate_answer: problem_text must be a non-empty string.")

        textbook_hits = self._retrieve_textbook_chunks(problem_text) if self._textbook_chunks else []
        merged_contexts = list(contexts) + textbook_hits

        rag_block = self._build_context_block(contexts, max_chars=self.max_context_chars)
        tb_block = self._build_context_block(textbook_hits, max_chars=self.textbook_max_chars)
        history_block = self._build_history_block(history)

        sys = self._system_prompt(subject=subject, grade_level=grade_level)
        style_instructions = self._style_instructions(style)

        prompt = f"""
You are helping an elementary-school student.

{style_instructions}

Conversation so far (may be empty):
{history_block}

Grounding rules:
- Use the provided CONTEXT blocks if helpful.
- Only cite when you truly used information from a chunk.
- If a sentence in FINAL_ANSWER uses a context chunk, wrap the WHOLE sentence like this:
  <cite chunk_id="chunk_id">That whole sentence goes here.</cite>
- If a step in STEPS uses a context chunk, wrap the WHOLE step like this:
  <cite chunk_id="chunk_id">That whole step goes here.</cite>
- Do NOT place citations only at the end of a sentence.
- Do NOT use [CITE:chunk_id].
- Do NOT cite sentences that do not use context.
- If one sentence/step uses one chunk, use one cite block.
- If no context was used for a sentence/step, leave it as plain text.
- In the CITATIONS section:
  - Output ONLY a bullet list.
  - Each line must be exactly "- <chunk_id>" OR "- NONE".
  - Do NOT write any natural language in CITATIONS.
- If no chunks were used, output exactly:
CITATIONS:
- NONE

Output format must be exactly:
FINAL_ANSWER:
...
STEPS:
...
CITATIONS:
- chunk_id
- chunk_id

QUESTION:
{problem_text.strip()}

CONTEXT (RAG):
{rag_block}

CONTEXT (TEXTBOOK_MD):
{tb_block}
""".strip()

        raw = self.gemini.generate_text(
            prompt,
            system_prompt=sys,
            model=self.text_model,
        )

        final_answer, steps, cites = self._parse_candidate_output(raw, merged_contexts)

        cites = [c for c in cites if c and c.upper() != "NONE"]

        return AnswerCandidate(
            candidate_id="A",
            style=style,
            final_answer=final_answer,
            steps=steps,
            citations=cites,
            raw_model_output=raw,
        )

  
    # Citation resolver 


    @staticmethod
    def describe_citation(chunk_id: str) -> Dict[str, str]:
        m = re.match(r"^TB:(.+):L(\d+)-L(\d+):#(\d+)$", chunk_id.strip())
        if m:
            fn, l1, l2, _ = m.group(1), m.group(2), m.group(3), m.group(4)
            return {"type": "textbook", "label": f"{fn} (lines {l1}-{l2})", "chunk_id": chunk_id}
        return {"type": "context", "label": chunk_id, "chunk_id": chunk_id}

  
    # Textbook loading
    

    def _load_textbook_md_path(self, md_path: str) -> List[ContextChunk]:
        if not os.path.isfile(md_path):
            raise ValueError(f"textbook_md_path not found: {md_path}")
        if not md_path.lower().endswith(".md"):
            raise ValueError(f"textbook_md_path must be a .md file: {md_path}")
        return self._chunk_markdown_file(md_path)

    def _load_textbook_md_dir(self, md_dir: str) -> List[ContextChunk]:
        if not os.path.isdir(md_dir):
            raise ValueError(f"textbook_md_dir not found or not a directory: {md_dir}")

        chunks: List[ContextChunk] = []
        for root, _, files in os.walk(md_dir):
            for fn in files:
                if fn.lower().endswith(".md"):
                    chunks.extend(self._chunk_markdown_file(os.path.join(root, fn)))
        return chunks

    def _chunk_markdown_file(self, path: str) -> List[ContextChunk]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8-sig") as f:
                lines = f.read().splitlines()

        filename = os.path.basename(path)

        blocks: List[Tuple[int, int, str]] = []
        start = 0
        buf: List[str] = []

        def flush(end_idx: int):
            nonlocal start, buf
            if not buf:
                return
            text = "\n".join(buf).strip()
            if text:
                blocks.append((start, end_idx, text))
            buf = []

        for i, line in enumerate(lines):
            if line.strip() == "":
                flush(i)
                start = i + 1
            else:
                buf.append(line)
        flush(len(lines))

        merged: List[Tuple[int, int, str]] = []
        i = 0
        while i < len(blocks):
            s, e, text = blocks[i]
            is_heading_only = bool(re.fullmatch(r"\s*#{1,6}\s+.+", text.strip()))
            if is_heading_only and i + 1 < len(blocks):
                s2, e2, t2 = blocks[i + 1]
                merged.append((s, e2, text.strip() + "\n" + t2.strip()))
                i += 2
            else:
                merged.append((s, e, text))
                i += 1

        out: List[ContextChunk] = []
        for idx, (s, e, text) in enumerate(merged, start=1):
            l1 = s + 1
            l2 = e
            chunk_id = f"TB:{filename}:L{l1}-L{l2}:#{idx}"
            out.append(
                ContextChunk(
                    chunk_id=chunk_id,
                    doc_id=filename,
                    content=text,
                    metadata={
                        "source": "textbook_md",
                        "file": filename,
                        "line_start": l1,
                        "line_end": l2,
                        "path": path,
                        "chunk_index": idx,
                    },
                    score=0.0,
                )
            )
        return out

    
    # Query enhancement
    

    def _enhance_query_for_textbook(self, query: str) -> str:
        q = (query or "").strip()
        q_low = q.lower()

        extra: List[str] = []

        frac_matches = re.findall(r"\b(\d+)\s*/\s*(\d+)\b", q_low)
        if frac_matches:
            extra.append("fraction fractions numerator denominator equivalent fractions")
            for a, b in frac_matches[:5]:
                extra.append(f"{a}/{b}")

        mixed_matches = re.findall(r"\b(\d+)\s+(\d+)\s*/\s*(\d+)\b", q_low)
        if mixed_matches:
            extra.append("mixed number improper fraction")
            for w, a, b in mixed_matches[:5]:
                extra.append(f"{w} {a}/{b}")

        if any(k in q_low for k in ["batch", "batches", "each", "per", "times", "product"]):
            extra.append("multiply multiplication times product")
        if any(k in q_low for k in ["total", "altogether", "sum", "in all", "combine"]):
            extra.append("add addition sum total")
        if any(k in q_low for k in ["left", "remain", "remaining", "difference"]):
            extra.append("subtract subtraction difference")
        if any(k in q_low for k in ["divide", "shared", "equal groups", "each group", "quotient"]):
            extra.append("divide division divisor dividend quotient remainder")
        if any(k in q_low for k in ["decimal", "$", "dollar", "cent"]):
            extra.append("decimal decimals money cents dollars")
        if any(k in q_low for k in ["area", "perimeter"]):
            extra.append("area perimeter square units")
        if "volume" in q_low:
            extra.append("volume cubic units")

        if not extra:
            return q
        return q + "\n\nTEXTBOOK_KEYWORDS: " + " ".join(extra)

    
    # Retrieval


    def _is_vocab_file(self, chunk: ContextChunk) -> bool:
        fname = (chunk.metadata.get("file") or chunk.doc_id or "").lower()
        return ("vocabulary" in fname) or ("glossary" in fname)

    def _retrieve_textbook_chunks(self, query: str) -> List[ContextChunk]:
        enhanced = self._enhance_query_for_textbook(query)
        q_tokens = self._tokenize(enhanced)
        if not q_tokens:
            return []

        scored_all: List[Tuple[float, ContextChunk]] = []

        for c in self._textbook_chunks:
            c_tokens = self._tokenize(c.content)
            if not c_tokens:
                continue

            score = self._overlap_score(q_tokens, c_tokens)

            if self._is_vocab_file(c):
                score *= self.vocab_boost

            scored_all.append((score, c))

        scored_all.sort(key=lambda x: x[0], reverse=True)

        scored = [(s, c) for (s, c) in scored_all if s >= self.textbook_min_score]
        if not scored:
            return []

        top_pairs: List[Tuple[float, ContextChunk]] = []
        k = self.textbook_top_k

        if self.reserve_vocab_slot:
            vocab_scored = [p for p in scored if self._is_vocab_file(p[1])]
            if vocab_scored:
                top_pairs.append(vocab_scored[0])
                k = max(0, k - 1)
                scored = [p for p in scored if p != vocab_scored[0]]

        top_pairs.extend(scored[:k])

        hits: List[ContextChunk] = []
        for s, c in top_pairs:
            hits.append(
                ContextChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    content=c.content,
                    metadata=dict(c.metadata),
                    score=float(s),
                )
            )
        return hits

    def debug_textbook_hits(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[ContextChunk]:
        if not getattr(self, "_textbook_chunks", None):
            return []

        old_k = self.textbook_top_k
        old_min = self.textbook_min_score
        try:
            self.textbook_top_k = top_k
            self.textbook_min_score = min_score
            return self._retrieve_textbook_chunks(query)
        finally:
            self.textbook_top_k = old_k
            self.textbook_min_score = old_min

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        t = (text or "").lower()

        frac_tokens = re.findall(r"\b\d+\s*/\s*\d+\b", t)
        frac_tokens = [x.replace(" ", "") for x in frac_tokens]

        tokens = re.findall(r"[a-z0-9]+", t)

        stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
            "is", "are", "was", "were", "be", "as", "at", "by", "from", "that",
            "this", "it", "you", "your", "we", "they", "i"
        }
        tokens = [x for x in tokens if x not in stop and len(x) >= 2]
        tokens.extend(frac_tokens)
        return tokens

    @staticmethod
    def _overlap_score(q_tokens: List[str], c_tokens: List[str]) -> float:
        qs = set(q_tokens)
        cs = set(c_tokens)
        inter = len(qs & cs)
        if inter == 0:
            return 0.0
        denom = (len(qs) * len(cs)) ** 0.5
        return float(inter / denom) if denom > 0 else 0.0


    # Prompt parts
  

    def _system_prompt(self, *, subject: str, grade_level: str) -> str:
        subject_line = f"Subject: {subject}" if subject else "Subject: (not specified)"
        grade_line = f"Grade: {grade_level}" if grade_level else "Grade: (not specified)"
        return f"""
You are a helpful tutor for elementary school students.
Be accurate, clear, and age-appropriate.
{subject_line}
{grade_line}
""".strip()

    def _style_instructions(self, style: AnswerStyle) -> str:
        if style == "brief":
            return """
Answer style: BRIEF
- Keep it short.
- Provide the final answer and 1-3 bullet steps max.
""".strip()
        return """
Answer style: DETAILED
- Explain step-by-step.
- Keep language simple and kid-friendly.
- Show intermediate steps clearly.
""".strip()

    # Context + history
   

    def _build_context_block(self, contexts: List[ContextChunk], max_chars: int) -> str:
        if not contexts:
            return "(No context)"

        blocks: List[str] = []
        total = 0
        for c in contexts:
            src = c.metadata.get("source", "rag")
            if src == "textbook_md":
                loc = f"{c.metadata.get('file', c.doc_id)} L{c.metadata.get('line_start')}-L{c.metadata.get('line_end')}"
                header = f"[{c.chunk_id}] (TEXTBOOK: {loc})"
            else:
                header = f"[{c.chunk_id}] (RAG)"

            piece = f"{header}\n{c.content}\n"
            if total + len(piece) > max_chars:
                break
            blocks.append(piece)
            total += len(piece)

        return "\n".join(blocks).strip() if blocks else "(Context truncated)"

    def _build_history_block(self, history: Optional[Sequence[Dict[str, str]]]) -> str:
        if not history:
            return "(No prior messages)"

        tail = list(history)[-20:]
        lines: List[str] = []
        for m in tail:
            role = (m.get("role") or "").upper()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role not in ("USER", "ASSISTANT", "SYSTEM"):
                role = "USER"
            lines.append(f"{role}: {content}")

        text = "\n".join(lines).strip()
        if len(text) <= self.history_max_chars:
            return text
        return "...\n" + text[-self.history_max_chars:]

    # -------------------------------
    # Output parsing (cite-block style)
    # -------------------------------

    def _parse_candidate_output(
        self,
        raw: str,
        contexts: List[ContextChunk],
    ) -> Tuple[str, str, List[str]]:
        text = (raw or "").strip()
        if not text:
            raise ValueError("Model output is empty.")

        def _get_section(label: str, next_labels: List[str]) -> str:
            start = text.find(label)
            if start < 0:
                return ""
            start += len(label)
            end_positions = [text.find(nl, start) for nl in next_labels]
            end_positions = [p for p in end_positions if p >= 0]
            end = min(end_positions) if end_positions else len(text)
            return text[start:end].strip()

        final_answer = _get_section("FINAL_ANSWER:", ["STEPS:", "CITATIONS:"])
        steps = _get_section("STEPS:", ["CITATIONS:"])
        cites_block = _get_section("CITATIONS:", [])

        citations: List[str] = []
        bad_format = False

        if cites_block:
            for line in cites_block.splitlines():
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("-"):
                    bad_format = True
                    continue
                cid = line[1:].strip()
                if not cid:
                    bad_format = True
                    continue
                citations.append(cid)
        else:
            citations = ["NONE"]

        if bad_format:
            citations = ["NONE"]

        inline = self._extract_inline_citations(text)
        if inline:
            citations = list(dict.fromkeys(inline + citations))

        if contexts:
            valid_ids = {c.chunk_id for c in contexts}
            citations = [cid for cid in citations if cid.upper() == "NONE" or cid in valid_ids]

        if not final_answer:
            final_answer = text
        if not steps:
            steps = "(No steps provided)"
        if not citations:
            citations = ["NONE"]

        return final_answer.strip(), steps.strip(), citations

    @staticmethod
    def _extract_inline_citations(text: str) -> List[str]:
        """
        Extract chunk ids from:
        <cite chunk_id="..."> ... </cite>
        """
        cites: List[str] = []
        pattern = r'<cite\s+chunk_id="([^"]+)">.*?</cite>'
        for m in re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE):
            cid = (m.group(1) or "").strip()
            if cid:
                cites.append(cid)
        return cites