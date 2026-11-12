"""Optional LLM backend with schema-constrained (Pydantic-validated) output.

When the server runs inside Claude Code, the *host* model already does the
final synthesis, so an internal LLM is optional. Without one, every LLM step
(extraction, grading, rewriting, generation, verification) falls back to a
deterministic implementation and the server acts as a verified-retrieval
engine. With one, those steps get sharper.

Supported providers:
  * anthropic  - ANTHROPIC_API_KEY
  * openai     - OPENAI_API_KEY (+ OPENAI_BASE_URL for any OpenAI-compatible
                 endpoint: Gemini's OpenAI endpoint, Ollama, vLLM, LM Studio...)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4.1-mini",
}


class LLM:
    def __init__(self, provider: str, model: str, client) -> None:
        self.provider = provider
        self.model = model
        self._client = client

    # ------------------------------------------------------------------
    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def structured(self, system: str, prompt: str, schema: type[T], max_tokens: int = 2000) -> T | None:
        """Ask for JSON matching `schema`; validate with Pydantic; one repair retry."""
        schema_json = json.dumps(schema.model_json_schema())
        sys = (
            f"{system}\n\nRespond with ONLY a JSON object that validates against this JSON Schema. "
            f"No prose, no code fences.\n{schema_json}"
        )
        last_err = ""
        for attempt in range(2):
            p = prompt if attempt == 0 else f"{prompt}\n\nYour previous output was invalid: {last_err}. Return valid JSON only."
            try:
                raw = self.complete(sys, p, max_tokens=max_tokens)
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as exc:
                last_err = str(exc)[:500]
            except Exception as exc:  # network / auth errors: degrade gracefully
                log.warning("LLM call failed (%s); falling back to heuristics", exc)
                return None
        log.warning("LLM structured output failed validation twice: %s", last_err)
        return None


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return raw[start : end + 1]


def build_llm(settings: Settings) -> LLM | None:
    provider = settings.llm_provider.lower()
    if provider == "none":
        return None
    if provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            return None
    model = settings.llm_model or _DEFAULT_MODELS.get(provider, "")
    try:
        if provider == "anthropic":
            import anthropic

            return LLM("anthropic", model, anthropic.Anthropic())
        if provider == "openai":
            import openai

            kwargs = {"base_url": settings.openai_base_url} if settings.openai_base_url else {}
            return LLM("openai", model, openai.OpenAI(**kwargs))
    except ImportError:
        log.warning("Provider %s selected but its SDK is not installed (pip install '.[%s]')", provider, provider)
        return None
    log.warning("Unknown LLM provider %r; running without an LLM", provider)
    return None
