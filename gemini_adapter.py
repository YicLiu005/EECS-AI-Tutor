# gemini_adapter.py
# Wrapper for Gemini API calls.

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import (
    GEMINI_API_KEY,
    GEMINI_API_BASE,
    GEMINI_TEXT_MODEL,
    GEMINI_VISION_MODEL,
    GEMINI_EMBED_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TOP_P,
    GEMINI_TOP_K,
)


class GeminiAPIError(RuntimeError):
    """Gemini API error."""
    pass


class GeminiAdapter:
    """Handle Gemini text, vision, and embedding requests."""
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        text_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        timeout_s: int = 60,
    ) -> None:
        self.api_key = (api_key or GEMINI_API_KEY).strip()
        self.api_base = (api_base or GEMINI_API_BASE).strip().rstrip("/")
        self.text_model = (text_model or GEMINI_TEXT_MODEL).strip()
        self.vision_model = (vision_model or GEMINI_VISION_MODEL).strip()
        self.embed_model = (embed_model or GEMINI_EMBED_MODEL).strip()
        self.timeout_s = int(timeout_s)

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is empty. Put it in config.py or set env GEMINI_API_KEY.")

    # --------------------------
    # Debug: list models
    # --------------------------
    def list_models(self) -> Dict[str, Any]:
        url = f"{self.api_base}/models"
        resp = requests.get(url, params={"key": self.api_key}, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise GeminiAPIError(f"list_models failed. HTTP {resp.status_code}: {resp.text}")
        return resp.json()

    # --------------------------
    # Text generation
    # --------------------------
    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        model: Optional[str] = None,
        temperature: float = GEMINI_TEMPERATURE,
        max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS,
        top_p: float = GEMINI_TOP_P,
        top_k: int = GEMINI_TOP_K,
    ) -> str:
        use_model = (model or self.text_model).strip()
        contents = self._build_contents_text(prompt, system_prompt=system_prompt)
        return self._call_generate_content(
            model_id=use_model,
            contents=contents,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
        )

    # --------------------------
    # Vision generation (image -> text/JSON)
    # --------------------------
    def vision_to_text(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
        task_hint: str = "",
        model: Optional[str] = None,
    ) -> str:
        use_model = (model or self.vision_model).strip()
        if not image_bytes:
            raise ValueError("vision_to_text: empty image_bytes")

        text_prompt = task_hint.strip() or """
Extract the question from the image.
Return ONLY valid JSON:
{
  "problem_text": "...",
  "options": [],
  "figure_desc": "",
  "constraints": [],
  "grade_level": "",
  "subject": ""
}
""".strip()

        contents = [{
            "role": "user",
            "parts": [
                {"text": text_prompt},
                {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("utf-8")}},
            ],
        }]

        return self._call_generate_content(
            model_id=use_model,
            contents=contents,
            temperature=0.2,
            max_output_tokens=1024,
            top_p=0.95,
            top_k=40,
        )

    # --------------------------
    # Embeddings
    # --------------------------
    def embed_text(self, texts: List[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("embed_text: texts must be a non-empty list of non-empty strings.")

        url = f"{self.api_base}/models/{self.embed_model}:embedContent"
        vectors: List[List[float]] = []

        for t in texts:
            payload = {
                "content": {"parts": [{"text": t}]},
                "taskType": task_type,
            }
            resp = requests.post(url, params={"key": self.api_key}, json=payload, timeout=self.timeout_s)
            if resp.status_code != 200:
                raise GeminiAPIError(f"embedContent failed. HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            emb = data.get("embedding", {}).get("values", None)
            if not isinstance(emb, list):
                raise GeminiAPIError(f"embedContent unexpected response: {data}")
            vectors.append([float(x) for x in emb])

        return vectors

    # --------------------------
    # JSON helper
    # --------------------------
    def try_parse_json(self, raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(raw, str):
            return None, "raw is not a string"
        s = raw.strip()
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, "no JSON braces found"
        snippet = s[start:end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj, None
            return None, "JSON parsed but not an object"
        except Exception as e:
            return None, f"json.loads failed: {e}"

    # --------------------------
    # Internal REST call
    # --------------------------
    def _call_generate_content(
        self,
        *,
        model_id: str,
        contents: List[Dict[str, Any]],
        temperature: float,
        max_output_tokens: int,
        top_p: float,
        top_k: int,
    ) -> str:
        url = f"{self.api_base}/models/{model_id}:generateContent"
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_output_tokens),
                "topP": float(top_p),
                "topK": int(top_k),
            },
        }
        resp = requests.post(url, params={"key": self.api_key}, json=payload, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise GeminiAPIError(
                f"Gemini API error. HTTP {resp.status_code}: {resp.text}\n"
                f"Tip: run list_models() to see supported models."
            )

        data = resp.json()
        try:
            candidates = data.get("candidates", [])
            parts = candidates[0]["content"]["parts"]
            texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
            out = "".join(texts).strip()
            return out if out else json.dumps(data, ensure_ascii=False)
        except Exception:
            return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _build_contents_text(prompt: str, *, system_prompt: str = "") -> List[Dict[str, Any]]:
        # Simple approach: prepend system prompt into user prompt
        merged = prompt if not system_prompt.strip() else (system_prompt.strip() + "\n\n" + prompt)
        return [{"role": "user", "parts": [{"text": merged}]}]