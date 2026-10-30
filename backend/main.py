"""FastAPI application: upload endpoints, job API, static web UI."""

from __future__ import annotations

import logging
import mimetypes
import random
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from . import media
from .comfy import ComfyClient, gpu_info
from .config import settings
from .engines import (
    EngineError,
    describe_all,
    discover,
    get_engine,
    invalidate_state,
    resolve_default,
    wan_frames,
)
from .store import store
from .worker import get_runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ogenjutsu")

STARTED_AT = time.time()
runner = get_runner(store)


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    store.load()
    store.purge_workdirs()
    runner.start()
    log.info(
        "OpenGenjutsu %s ready (engine=%s, data=%s)",
        settings.version,
        resolve_default(),
        settings.data_dir,
    )
    yield
    runner.stop()


app = FastAPI(title="OpenGenjutsu", version=settings.version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length", "Accept-Ranges"],
)


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #


def require_auth(
    request: Request,
    token: Annotated[str | None, Query()] = None,
) -> None:
    """Shared-secret gate for the API. No-op when OG_TOKEN is empty."""
    expected = settings.auth_token
    if not expected:
        return
    supplied = request.headers.get("x-auth-token") or token or ""
    if supplied != expected:
        raise HTTPException(status_code=401, detail="invalid or missing access token")


Auth = Annotated[None, Depends(require_auth)]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _max_bytes() -> int:
    return max(1, settings.max_upload_mb) * 1024 * 1024


async def _save_upload(upload: UploadFile, dest: Path) -> int:
    """Stream an upload to disk with a hard size cap."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    limit = _max_bytes()
    total = 0
    try:
        with open(dest, "wb") as handle:
            while True:
                chunk = await upload.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds the {settings.max_upload_mb} MB limit",
                    )
                handle.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"could not store upload: {exc}") from exc

    if total == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    return total


def _safe_suffix(filename: str | None, allowed: set[str], fallback: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in allowed else fallback


def _snap(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _media_type(path: Path, fallback: str = "application/octet-stream") -> str:
    """Content type from the file extension (uploaded files keep their suffix)."""
    overrides = {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".gif": "image/gif",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return overrides.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or fallback


def _file(path: Path, media_type: str, filename: str | None = None) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not available")
    headers = {"Cache-Control": "no-store"}
    if filename:
        headers["Content-Disposition"] = f'inline; filename="{filename}"'
    return FileResponse(path, media_type=media_type, headers=headers)


def _job_or_404(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


# --------------------------------------------------------------------------- #
# meta endpoints
# --------------------------------------------------------------------------- #


@app.get("/api/health", dependencies=[Depends(require_auth)])
def health() -> dict[str, Any]:
    comfy = ComfyClient()
    stats = comfy.system_stats()
    disk = shutil.disk_usage(settings.data_dir)
    engines = describe_all()
    return {
        "ok": True,
        "version": settings.version,
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "started_at": STARTED_AT,
        "engine_default": resolve_default(),        "comfy": {
            "url": settings.comfy_url,
            "reachable": stats is not None,
            "system": (stats or {}).get("system") if stats else None,
            "devices": (stats or {}).get("devices") if stats else None,
            "staging_mode": "filesystem" if settings.comfy_input_dir else "http-upload",
        },
        "ffmpeg": media.ffmpeg_status(),
        "gpu": gpu_info(),
        "queue": {
            "queued": len(store.queued_ids()),
            "active": next(
                (j.id for j in store.all() if j.status == "running"), None
            ),
        },
        "storage": {
            "data_dir": str(settings.data_dir),
            "free_mb": disk.free // (1024 * 1024),
            "total_mb": disk.total // (1024 * 1024),
        },
        "engines": engines,
        "auth": {"required": bool(settings.auth_token)},
    }


@app.get("/api/engines", dependencies=[Depends(require_auth)])
def engines() -> dict[str, Any]:
    return {"default": resolve_default(), "engines": describe_all()}


@app.post("/api/engines/reload", dependencies=[Depends(require_auth)])
def reload_engines() -> dict[str, Any]:
    invalidate_state()
    discover(force=True)
    return {"default": resolve_default(), "engines": describe_all()}


@app.get("/", response_class=HTMLResponse)
def index() -> Response:
    page = settings.frontend_dir / "index.html"
    if not page.is_file():
        return HTMLResponse(
            "<h1>OpenGenjutsu</h1><p>Frontend not found at "
            f"<code>{settings.frontend_dir}</code>.</p>",
            status_code=500,
        )
    return HTMLResponse(page.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@app.get("/app.js")
def app_js() -> Response:
    return _file(settings.frontend_dir / "app.js", "application/javascript")


@app.get("/app.css")
def app_css() -> Response:
    return _file(settings.frontend_dir / "app.css", "text/css")


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #


@app.post("/api/jobs", dependencies=[Depends(require_auth)], status_code=201)
async def create_job(
    motion: Annotated[UploadFile, File(description="motion / driving video")],
    character: Annotated[UploadFile, File(description="character reference image")],
    engine: Annotated[str | None, Form()] = None,
    prompt: Annotated[str, Form()] = "",
    negative: Annotated[str, Form()] = "",
    width: Annotated[int, Form()] = 832,
    height: Annotated[int, Form()] = 480,
    fps: Annotated[float, Form()] = 16.0,
    seconds: Annotated[float, Form()] = 4.0,
    steps: Annotated[int, Form()] = 6,
    cfg: Annotated[float, Form()] = 1.0,
    shift: Annotated[float, Form()] = 5.0,
    seed: Annotated[int, Form()] = -1,
    keep_audio: Annotated[bool, Form()] = True,
    audio_volume: Annotated[float, Form()] = 1.0,
) -> dict[str, Any]:
    engine_id = engine or resolve_default()
    try:
        impl = get_engine(engine_id)
    except EngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state = impl.preflight()
    if not state["ready"]:
        raise HTTPException(
            status_code=409,
            detail={"message": "engine not ready", "problems": state["problems"]},
        )

    defaults = impl.defaults()
    max_seconds = float(defaults.get("max_seconds") or settings.default_max_seconds)

    width = _snap(int(width or defaults["width"]), 16)
    height = _snap(int(height or defaults["height"]), 16)
    fps = max(1.0, min(60.0, float(fps or defaults["fps"])))
    requested_seconds = max(0.5, float(seconds or 4.0))
    seconds = min(max_seconds, requested_seconds)
    if requested_seconds > max_seconds:
        # One API job is one Wan window. The UI caps the request too; keeping the
        # explicit clamp here prevents an oversized latent from reaching ComfyUI.
        pass
    # Wan Animate's conditioning and checkpoint are trained at 16 fps.  Keep
    # sampling at that native rate, then deliver 24/30/60 fps by interpolation
    # after generation.  This avoids pretending that 60 fps gives 60 distinct
    # diffusion frames and keeps the compute cost bounded.
    model_fps = min(16.0, fps)
    frames = wan_frames(int(round(seconds * model_fps)))
    steps = max(1, min(60, int(steps or defaults["steps"])))
    cfg = max(0.0, min(20.0, float(cfg if cfg is not None else defaults["cfg"])))
    shift = max(0.0, min(20.0, float(shift)))
    seed = int(seed)
    if seed < 0:
        seed = random.randint(1, 2**31 - 1)
    audio_volume = max(0.0, min(2.0, float(audio_volume)))

    job = store.create(
        params={
            "prompt": prompt.strip(),
            "negative": negative.strip(),
            "width": width,
            "height": height,
            "fps": fps,
            "model_fps": model_fps,
            "frames": frames,
            "seconds": round(frames / model_fps, 3),
            "steps": steps,
            "cfg": cfg,
            "shift": shift,
            "seed": seed,
            "keep_audio": bool(keep_audio),
            "audio_volume": audio_volume,
        },
        engine=engine_id,
    )

    try:
        motion_name = "motion" + _safe_suffix(
            motion.filename, {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".gif"}, ".mp4"
        )
        char_name = "character" + _safe_suffix(
            character.filename, {".png", ".jpg", ".jpeg", ".webp", ".bmp"}, ".png"
        )
        await _save_upload(motion, job.input_dir / motion_name)
        await _save_upload(character, job.input_dir / char_name)

        motion_path = job.input_dir / motion_name
        char_path = job.input_dir / char_name

        info = media.probe(motion_path)
        if not info["has_video"]:
            raise HTTPException(status_code=400, detail="motion file has no video stream")
        if info["duration"] and info["duration"] < 0.3:
            raise HTTPException(status_code=400, detail="motion video is too short")

        try:
            char_info = media.probe(char_path)
        except media.MediaError:
            char_info = {"width": 0, "height": 0}
        if not char_info.get("width"):
            raise HTTPException(status_code=400, detail="character reference is not a readable image")

        # Extract audio once, at upload time: the toggle then works instantly,
        # and the source's audio is preserved exactly (no re-encode of the video).
        has_audio = bool(info["has_audio"])
        if has_audio and keep_audio:
            try:
                media.extract_audio(motion_path, job.input_dir / "audio.wav")
            except media.MediaError as exc:
                store.append_log(job.id, f"audio extraction failed: {exc}")
                has_audio = False

        store.update(
            job.id,
            params={**job.params, "motion_file": motion_name, "character_file": char_name},
            source_info={
                "duration": round(float(info["duration"]), 3),
                "width": info["width"],
                "height": info["height"],
                "fps": round(info["fps"], 3),
                "frames": info["nb_frames"],
                "codec": info["video_codec"],
                "audio_codec": info["audio_codec"],
                "size_bytes": info["size_bytes"],
            },
            character_info={
                "width": char_info.get("width", 0),
                "height": char_info.get("height", 0),
                "format": char_info.get("format_name", ""),
            },
            has_source_audio=has_audio,
        )
        store.append_log(
            job.id,
            f"sources accepted: motion {info['width']}x{info['height']} "
            f"{info['duration']:.2f}s audio={'yes' if has_audio else 'no'}, "
            f"character {char_info.get('width')}x{char_info.get('height')}",
        )
    except HTTPException as exc:
        store.update(
            job.id,
            status="error",
            stage="rejected",
            error=str(exc.detail),
            finished=time.time(),
        )
        raise
    except media.MediaError as exc:
        store.update(
            job.id, status="error", stage="rejected", error=str(exc), finished=time.time()
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    runner.submit(job)
    return store.api_one(job.id) or {}


@app.get("/api/jobs", dependencies=[Depends(require_auth)])
def list_jobs(limit: int = 30) -> dict[str, Any]:
    return {"jobs": store.api_list(limit=max(1, min(200, limit)))}


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_auth)])
def get_job(job_id: str) -> dict[str, Any]:
    payload = store.api_one(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="job not found")
    return payload


@app.get("/api/jobs/{job_id}/result", dependencies=[Depends(require_auth)])
def job_result(job_id: str, download: int = 0) -> Response:
    job = _job_or_404(job_id)
    path = job.result_path
    if not path:
        raise HTTPException(status_code=404, detail="no render yet")
    if download:
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f'attachment; filename="ogenjutsu-{job.id}.mp4"'
            },
        )
    return _file(path, "video/mp4")


@app.get("/api/jobs/{job_id}/source", dependencies=[Depends(require_auth)])
def job_source(job_id: str) -> Response:
    job = _job_or_404(job_id)
    path = job.motion_path
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="source not available")
    return _file(path, _media_type(path, "video/mp4"))


@app.get("/api/jobs/{job_id}/character", dependencies=[Depends(require_auth)])
def job_character(job_id: str) -> Response:
    job = _job_or_404(job_id)
    path = job.character_path
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="character not available")
    return _file(path, _media_type(path, "image/png"))


@app.get("/api/jobs/{job_id}/thumb", dependencies=[Depends(require_auth)])
def job_thumb(job_id: str) -> Response:
    job = _job_or_404(job_id)
    path = job.thumb_path
    if not path:
        raise HTTPException(status_code=404, detail="no thumbnail")
    return _file(path, "image/jpeg")


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_auth)])
def cancel_job(job_id: str) -> dict[str, Any]:
    _job_or_404(job_id)
    ok = runner.cancel(job_id)
    return {"canceled": ok, "job": store.api_one(job_id)}


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(require_auth)])
def delete_job(job_id: str) -> dict[str, Any]:
    _job_or_404(job_id)
    runner.cancel(job_id)
    return {"deleted": store.delete(job_id)}


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #


@app.post("/api/selftest", dependencies=[Depends(require_auth)])
def selftest(engine: str | None = None) -> dict[str, Any]:
    """Render a synthetic clip end-to-end, including the audio toggle.

    Validates ffmpeg, the queue, the engine and audio muxing with zero user input.
    """
    if not media.ffmpeg_status()["ok"]:
        raise HTTPException(status_code=409, detail=media.ffmpeg_status()["error"])

    engine_id = engine or "mock"
    try:
        impl = get_engine(engine_id)
    except EngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state = impl.preflight()
    if not state["ready"]:
        raise HTTPException(status_code=409, detail=state["problems"])

    job = store.create(
        params={
            "prompt": "self test",
            "negative": "",
            "width": 512,
            "height": 320,
            "fps": 16.0,
            "frames": 49,
            "seconds": round(49 / 16, 3),
            "steps": 1,
            "cfg": 1.0,
            "shift": 5.0,
            "seed": 1234,
            "keep_audio": True,
            "audio_volume": 1.0,
            "selftest": True,
        },
        engine=engine_id,
    )

    tmp = job.input_dir
    try:
        clip = media.make_test_clip(
            tmp / "raw_motion.mp4", seconds=3.0, width=512, height=320, fps=16
        )
        tone = media.make_test_tone(tmp / "tone.wav", seconds=3.0)
        motion = tmp / "motion.mp4"
        media.mux_audio(clip, tone, motion, fade_out=0.2)
        media.make_test_image(tmp / "character.png", 512, 320)

        info = media.probe(motion)
        store.update(
            job.id,
            params={
                **job.params,
                "motion_file": "motion.mp4",
                "character_file": "character.png",
            },
            source_info={
                "duration": round(float(info["duration"]), 3),
                "width": info["width"],
                "height": info["height"],
                "fps": round(info["fps"], 3),
                "frames": info["nb_frames"],
                "codec": info["video_codec"],
                "audio_codec": info["audio_codec"],
                "size_bytes": info["size_bytes"],
            },
            character_info={"width": 512, "height": 320, "format": "png"},
            has_source_audio=True,
        )
        media.extract_audio(motion, job.input_dir / "audio.wav")
    except media.MediaError as exc:
        store.update(
            job.id, status="error", stage="rejected", error=str(exc), finished=time.time()
        )
        raise HTTPException(status_code=500, detail=f"self-test setup failed: {exc}") from exc

    store.append_log(job.id, "self-test: synthetic motion + tone generated")
    runner.submit(job)
    return {"job": store.api_one(job.id), "engine": engine_id}


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #


@app.exception_handler(EngineError)
async def engine_error_handler(request: Request, exc: EngineError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
