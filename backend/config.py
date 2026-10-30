"""Runtime configuration, read from environment variables (.env is loaded by run.py)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.0.0"


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Settings:
    """Process-wide settings. Call refresh() to re-read the environment."""

    def __init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self.project_root = PROJECT_ROOT
        self.version = VERSION

        self.frontend_dir = Path(
            os.getenv("OG_FRONTEND_DIR", str(PROJECT_ROOT / "frontend"))
        )
        self.data_dir = Path(
            os.getenv("OG_DATA_DIR", "").strip() or str(PROJECT_ROOT / "data")
        )
        self.jobs_dir = self.data_dir / "jobs"
        self.tmp_dir = self.data_dir / "tmp"
        self.logs_dir = self.data_dir / "logs"

        self.host = os.getenv("OG_HOST", "0.0.0.0")
        self.port = env_int("OG_PORT", 7860)

        self.auth_token = os.getenv("OG_TOKEN", "").strip()

        raw_cors = os.getenv("OG_CORS", "*").strip() or "*"
        self.cors_origins = ["*"] if raw_cors == "*" else [
            o.strip() for o in raw_cors.split(",") if o.strip()
        ]

        self.comfy_url = os.getenv("OG_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
        raw_input = os.getenv("OG_COMFY_INPUT_DIR", "").strip()
        self.comfy_input_dir = Path(raw_input) if raw_input else None

        self.engine = os.getenv("OG_ENGINE", "auto").strip() or "auto"

        self.keep_jobs = env_int("OG_KEEP_JOBS", 40)
        self.max_upload_mb = env_int("OG_MAX_UPLOAD_MB", 512)

        # Skip the manifest VRAM gate (try a too-big model anyway).
        self.ignore_vram = env_bool("OG_IGNORE_VRAM", False)

        # Inference defaults (the UI sends its own values; these are fallbacks).
        self.default_width = env_int("OG_DEFAULT_WIDTH", 832)
        self.default_height = env_int("OG_DEFAULT_HEIGHT", 480)
        self.default_fps = env_float("OG_DEFAULT_FPS", 16.0)
        self.default_max_seconds = env_float("OG_DEFAULT_MAX_SECONDS", 8.0)
        self.job_timeout_s = env_int("OG_JOB_TIMEOUT", 3600)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.jobs_dir, self.tmp_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
