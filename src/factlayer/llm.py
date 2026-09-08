"""Provider-agnostic LLM client with a content-addressed disk cache.

Two things matter here and both are about working on a free tier.

1. Every call is keyed by a hash of (model, prompt, schema) and cached to disk.
   Re-running the pipeline after a code change costs nothing and hits no rate
   limit. This is the difference between iterating freely and spending the
   evening asleep behind HTTP 429s.

2. The model is asked for strict JSON. If it returns something unparseable we
   retry once with the validation error fed back to it, then give up and record
   the failure rather than inventing a result.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import settings

log = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    """Cheap running tally so the README can quote real numbers."""

    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "failures": self.failures,
        }


def _extract_json(text: str) -> Any:
    """Pull a JSON object out of a model response.

    Reasoning-style models like gpt-oss wrap their answer in prose or fences
    more often than the API contract suggests, so we degrade gracefully:
    straight parse, then fenced block, then the outermost brace pair.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise LLMError(f"could not parse JSON from response: {text[:300]}")


class LLMClient:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = Usage()
        self._sem = asyncio.Semaphore(settings.llm_max_concurrency)
        self._client: httpx.AsyncClient | None = None

    # -- cache ------------------------------------------------------------

    def _key(self, system: str, user: str, temperature: float) -> str:
        blob = json.dumps(
            [settings.llm_model, system, user, temperature], sort_keys=True
        ).encode()
        return hashlib.sha256(blob).hexdigest()

    def _cache_path(self, key: str) -> Path:
        # Shard by prefix so the directory stays navigable across thousands of calls.
        d = self.cache_dir / key[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json"

    def _read_cache(self, key: str) -> Any | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())["response"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _write_cache(self, key: str, system: str, user: str, response: Any) -> None:
        payload = {
            "model": settings.llm_model,
            "system": system,
            "user": user,
            "response": response,
        }
        try:
            self._cache_path(key).write_text(json.dumps(payload, indent=2))
        except OSError as exc:  # pragma: no cover - disk problems shouldn't be fatal
            log.warning("cache write failed: %s", exc)

    # -- transport --------------------------------------------------------

    async def _post(self, system: str, user: str, temperature: float) -> str:
        if not settings.llm_api_key:
            raise LLMError(
                "No API key. Copy .env.example to .env and set FACTLAYER_LLM_API_KEY."
            )
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=settings.llm_timeout_s)

        body = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
        url = settings.llm_base_url.rstrip("/") + "/chat/completions"

        # Free tiers rate-limit aggressively; back off rather than fail the run.
        delay = 2.0
        last: Exception | None = None
        for attempt in range(5):
            try:
                r = await self._client.post(url, json=body, headers=headers)
                if r.status_code == 429 or r.status_code >= 500:
                    retry_after = float(r.headers.get("retry-after", delay))
                    await asyncio.sleep(min(retry_after, 30) + random.uniform(0, 1))
                    delay = min(delay * 2, 30)
                    continue
                r.raise_for_status()
                data = r.json()
                self.usage.calls += 1
                if (u := data.get("usage")):
                    self.usage.prompt_tokens += u.get("prompt_tokens", 0)
                    self.usage.completion_tokens += u.get("completion_tokens", 0)
                return data["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last = exc
                await asyncio.sleep(delay + random.uniform(0, 1))
                delay = min(delay * 2, 30)
        raise LLMError(f"LLM request failed after retries: {last}")

    # -- public API -------------------------------------------------------

    async def json_call(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        validate=None,
    ) -> Any:
        """Return parsed JSON, using the cache when possible.

        `validate` is an optional callable that raises on a bad shape; when it
        does, we retry once with the error appended so the model can repair its
        own output.
        """
        key = self._key(system, user, temperature)
        if (hit := self._read_cache(key)) is not None:
            self.usage.cache_hits += 1
            return hit

        async with self._sem:
            raw = await self._post(system, user, temperature)

        try:
            parsed = _extract_json(raw)
            if validate:
                validate(parsed)
        except Exception as exc:
            repair = (
                f"{user}\n\n"
                f"Your previous reply could not be used. Error: {exc}\n"
                f"Reply with valid JSON only, matching the requested shape exactly."
            )
            async with self._sem:
                raw2 = await self._post(system, repair, temperature)
            try:
                parsed = _extract_json(raw2)
                if validate:
                    validate(parsed)
            except Exception as exc2:
                self.usage.failures += 1
                raise LLMError(f"model produced unusable JSON twice: {exc2}") from exc2

        self._write_cache(key, system, user, parsed)
        return parsed

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
