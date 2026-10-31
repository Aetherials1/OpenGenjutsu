"""Engine registry: discovers workflow templates in ``backend/workflows/``."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..config import settings
from .base import (
    Engine,
    EngineError,
    EngineResult,
    JobSpec,
    TemplateEngine,
    inject,
    placeholder_values,
    wan_frames,
)
from .mock import MockEngine

log = logging.getLogger("ogenjutsu.engines")

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"
_engines: dict[str, Engine] | None = None
# engine_id -> (timestamp, preflight state). Probing ComfyUI takes ~2s when it is
# down, and the UI polls health often, so results are cached briefly.
_state_cache: dict[str, tuple[float, dict]] = {}
STATE_TTL = 8.0


def _manifest_priority(directory: Path) -> int:
    try:
        data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        return int(data.get("priority", 100))
    except (OSError, ValueError, TypeError):
        return 100


def discover(force: bool = False) -> dict[str, Engine]:
    global _engines
    if _engines is not None and not force:
        return _engines

    engines: dict[str, Engine] = {"mock": MockEngine()}
    if WORKFLOW_DIR.is_dir():
        for directory in sorted(
            WORKFLOW_DIR.iterdir(), key=lambda d: (_manifest_priority(d), d.name)
        ):
            if not directory.is_dir():
                continue
            if not (directory / "manifest.json").is_file():
                continue
            if not (directory / "template.json").is_file():
                continue
            try:
                engine = TemplateEngine(directory)
            except Exception as exc:  # malformed template should not kill the app
                log.warning("skipping workflow %s: %s", directory.name, exc)
                continue
            engines[engine.id] = engine

    _engines = engines
    return engines


def all_engines() -> list[Engine]:
    return list(discover().values())


def engine_state(engine_id: str, *, force: bool = False, ttl: float = STATE_TTL) -> dict:
    """Cached preflight for one engine."""
    engine = get_engine(engine_id)
    now = time.time()
    hit = _state_cache.get(engine.id)
    if hit and not force and now - hit[0] < ttl:
        return hit[1]
    try:
        state = engine.preflight()
    except Exception as exc:  # noqa: BLE001 - a broken engine must not 500 health
        state = {"ready": False, "problems": [f"preflight failed: {exc}"], "warnings": []}
    _state_cache[engine.id] = (now, state)
    return state


def invalidate_state() -> None:
    _state_cache.clear()


def get_engine(engine_id: str | None) -> Engine:
    engines = discover()
    if engine_id and engine_id in engines:
        return engines[engine_id]
    if engine_id and engine_id not in engines:
        raise EngineError(f"unknown engine '{engine_id}'")
    default = resolve_default()
    if not default:
        raise EngineError(
            "no Wan Animate replacement engine is ready; start ComfyUI and install "
            "the Wan 2.2 Animate Mix workflow/models"
        )
    return engines[default]


def resolve_default() -> str:
    """Pick a ready replacement workflow; never silently choose pose-only/mock."""
    configured = settings.engine
    engines = discover()
    if configured and configured != "auto":
        if configured in engines:
            return configured
        log.warning("configured engine '%s' not found, falling back", configured)

    # A pose-only Animate 2 graph does not remove the source performer.  It is
    # useful as an explicit fallback, but must not win automatic replacement.
    ordered = sorted(
        (
            e for e in engines.values()
            if e.requires_comfy and getattr(e, "replacement_mode", "placeholder") == "sam2_mix"
        ),
        key=lambda e: _manifest_priority(Path(e.dir)) if hasattr(e, "dir") else 100,
    )
    for engine in ordered:
        try:
            if engine_state(engine.id)["ready"]:
                return engine.id
        except Exception:
            continue

    # Do not report a placeholder composite as a successful replacement.  The
    # caller can still explicitly select mock or pose-only for diagnostics, but
    # normal auto mode must make the missing replacement setup visible.
    return ""


def describe_all() -> list[dict]:
    """Engine cards for the UI. Preflights run concurrently so health stays snappy."""
    engines = all_engines()
    states: dict[str, dict] = {}

    def probe(engine: Engine) -> tuple[str, dict]:
        try:
            return engine.id, engine_state(engine.id)
        except Exception as exc:  # noqa: BLE001
            return engine.id, {
                "ready": False,
                "problems": [str(exc)],
                "warnings": [],
            }

    if engines:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(4, len(engines))) as pool:
            for engine_id, state in pool.map(probe, engines):
                states[engine_id] = state

    out: list[dict] = []
    for engine in engines:
        try:
            out.append(engine.info(states.get(engine.id)))
        except Exception as exc:
            out.append(
                {
                    "id": getattr(engine, "id", "?"),
                    "label": getattr(engine, "label", "?"),
                    "ready": False,
                    "problems": [str(exc)],
                    "warnings": [],
                    "defaults": {},
                }
            )
    return out


__all__ = [
    "Engine",
    "EngineError",
    "EngineResult",
    "JobSpec",
    "MockEngine",
    "TemplateEngine",
    "inject",
    "placeholder_values",
    "wan_frames",
    "all_engines",
    "describe_all",
    "discover",
    "engine_state",
    "get_engine",
    "invalidate_state",
    "resolve_default",
]
