"""Configuration, loaded from the environment (never from the repository)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    # Any OpenAI-compatible endpoint works. Default is Groq's free tier, which
    # means a reviewer can run this project without a paid account.
    llm_base_url: str = os.getenv("FACTLAYER_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_model: str = os.getenv("FACTLAYER_LLM_MODEL", "openai/gpt-oss-120b")
    llm_api_key: str = os.getenv("FACTLAYER_LLM_API_KEY", "")
    llm_max_concurrency: int = int(os.getenv("FACTLAYER_LLM_MAX_CONCURRENCY", "4"))
    llm_timeout_s: float = float(os.getenv("FACTLAYER_LLM_TIMEOUT_S", "120"))

    db_path: Path = _path("FACTLAYER_DB_PATH", "data/factlayer.db")
    cache_dir: Path = _path("FACTLAYER_CACHE_DIR", "data/cache")
    upload_dir: Path = _path("FACTLAYER_UPLOAD_DIR", "data/uploads")

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
