#!/usr/bin/env python
"""Launcher for the OpenGenjutsu backend.

Usage:
    python run.py                # default host/port from .env (0.0.0.0:7860)
    python run.py --reload       # auto-restart on code changes (dev)
    python run.py --port 8000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="OpenGenjutsu backend")
    parser.add_argument("--host", default=os.getenv("OG_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("OG_PORT", "7860")))
    parser.add_argument("--reload", action="store_true", help="reload on file changes")
    args = parser.parse_args()

    import uvicorn

    print(f"OpenGenjutsu -> http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(ROOT / "backend")] if args.reload else None,
        log_level="info",
    )


if __name__ == "__main__":
    main()
