"""Configuration, loaded from the environment (never from the repository)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


def _api_key() -> str:
    """Read the API key, tolerating the name the provider's own docs use.

    People (and reviewers) tend to already have GROQ_API_KEY or OPENAI_API_KEY
    exported. Accepting those avoids a silent "no key" failure that looks like a
    bug in the pipeline.
    """
    for name in (
        "FACTLAYER_LLM_API_KEY",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "CEREBRAS_API_KEY",
    ):
        if (v := os.getenv(name, "").strip().strip('"').strip("'")):
            return v
    return ""


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    # Any OpenAI-compatible endpoint works. Default is Groq's free tier, which
    # means a reviewer can run this project without a paid account.
    llm_base_url: str = os.getenv("FACTLAYER_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_model: str = os.getenv("FACTLAYER_LLM_MODEL", "openai/gpt-oss-120b")
    llm_api_key: str = _api_key()
    llm_max_concurrency: int = int(os.getenv("FACTLAYER_LLM_MAX_CONCURRENCY", "4"))
    llm_timeout_s: float = float(os.getenv("FACTLAYER_LLM_TIMEOUT_S", "120"))
    # Generous enough that a dense page's claim list is never cut short.
    llm_max_tokens: int = int(os.getenv("FACTLAYER_LLM_MAX_TOKENS", "16000"))
    # "low" | "medium" | "high" | "" to omit. Extraction is copying, not deduction.
    llm_reasoning_effort: str = os.getenv("FACTLAYER_LLM_REASONING_EFFORT", "low")

    db_path: Path = _path("FACTLAYER_DB_PATH", "data/factlayer.db")
    cache_dir: Path = _path("FACTLAYER_CACHE_DIR", "data/cache")
    upload_dir: Path = _path("FACTLAYER_UPLOAD_DIR", "data/uploads")

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
