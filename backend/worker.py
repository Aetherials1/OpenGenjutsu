"""Single background worker that renders queued jobs one at a time.

One worker is deliberate: a free Colab T4 can only hold one diffusion model at a
time, and serialising also makes progress/ETA meaningful and cancellation safe.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import traceback
from pathlib import Path

from .config import settings
from .engines import EngineError, JobSpec, get_engine
from .media import (
    MediaError,
    mux_audio,
    prepare_character,
    prepare_motion,
    probe,
    retime_video,
    strip_audio,
    thumbnail,
)
from .store import Job, JobStore

log = logging.getLogger("ogenjutsu.worker")


class JobRunner:
    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._queue: queue.Queue[str] = queue.Queue()
        self._cancel: dict[str, threading.Event] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ogenjutsu-worker", daemon=True)
        self._thread.start()
        log.info("worker started")

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait("__stop__")
        except queue.Full:
            pass

    def submit(self, job: Job) -> None:
        self._cancel[job.id] = threading.Event()
        self._queue.put(job.id)
        self.store.append_log(
            job.id, f"queued for {job.engine} ({self._queue.qsize()} in queue)"
        )

    def cancel(self, job_id: str) -> bool:
        event = self._cancel.get(job_id)
        job = self.store.get(job_id)
        if not job:
            return False
        if event:
            event.set()
        if job.status == "queued":
            self.store.update(
                job_id,
                status="canceled",
                stage="canceled",
                finished=time.time(),
                error=None,
                log="canceled while queued",
            )
            return True
        if job.status == "running":
            self.store.append_log(job_id, "cancel requested")
            return True
        return False

    def is_canceled(self, job_id: str) -> bool:
        event = self._cancel.get(job_id)
        return bool(event and event.is_set())

    # ------------------------------------------------------------------- inner
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job_id == "__stop__":
                break
            try:
                self._process(job_id)
            except Exception:  # never let the worker die
                log.exception("unhandled error in job %s", job_id)
                self.store.update(
                    job_id,
                    status="error",
                    stage="failed",
                    error="internal error - see server logs",
                    finished=time.time(),
                )
            finally:
                self._queue.task_done()
                self._cancel.pop(job_id, None)
                shutil.rmtree(
                    settings.jobs_dir / job_id / "work", ignore_errors=True
                )
                self._prune()

    def _prune(self) -> None:
        try:
            self.store.prune()
        except Exception:
            log.debug("prune failed", exc_info=True)

    def _process(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if not job:
            return
        if job.status == "canceled":
            return

        self.store.update(
            job_id,
            status="running",
            started=time.time(),
            stage="starting",
            progress=0.01,
        )
        self.store.append_log(
            job_id,
            f"start engine={job.engine} {job.params.get('width')}x{job.params.get('height')} "
            f"{job.params.get('frames')}f @{job.params.get('fps')}fps",
        )

        try:
            result = self._render(job)
        except EngineError as exc:
            if "canceled" in str(exc).lower():
                self.store.update(
                    job_id,
                    status="canceled",
                    stage="canceled",
                    finished=time.time(),
                    error=None,
                )
                self.store.append_log(job_id, "canceled")
            else:
                self.store.update(
                    job_id,
                    status="error",
                    stage="failed",
                    error=str(exc),
                    finished=time.time(),
                )
                self.store.append_log(job_id, f"ERROR {exc}")
            return
        except MediaError as exc:
            self.store.update(
                job_id, status="error", stage="failed", error=str(exc), finished=time.time()
            )
            self.store.append_log(job_id, f"ERROR {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - surface everything to the UI
            log.exception("job %s failed", job_id)
            self.store.update(
                job_id,
                status="error",
                stage="failed",
                error=f"{type(exc).__name__}: {exc}",
                finished=time.time(),
            )
            self.store.append_log(job_id, traceback.format_exc(limit=3))
            return

        # ------------------------------------------------------------- finalise
        job = self.store.get(job_id)
        if not job:
            return
        try:
            out_path, applied = self._finalise_audio(job, result.path)
            meta = dict(result.meta)
            info = probe(out_path)
            meta.update(
                {
                    "width": info["width"],
                    "height": info["height"],
                    "fps": round(info["fps"], 3),
                    "duration": round(float(info["duration"]), 3),
                    "frames": info["nb_frames"],
                    "has_audio": bool(info["has_audio"]),
                    "bytes": out_path.stat().st_size,
                    "render_meta": result.meta,
                }
            )
            thumb = thumbnail(out_path, job.output_dir / "thumb.jpg", at=min(0.7, max(0.1, info["duration"] * 0.1)))
            self.store.update(
                job_id,
                status="done",
                stage="done",
                progress=1.0,
                finished=time.time(),
                meta=meta,
                audio_applied=applied,
                error=None,
            )
            self.store.append_log(
                job_id,
                f"done in {round((self.store.get(job_id) or job).elapsed or 0, 1)}s -> "
                f"{meta['width']}x{meta['height']} {meta['duration']}s "
                f"audio={'original' if applied else 'none'}",
            )
            if thumb is None:
                self.store.append_log(job_id, "warning: thumbnail generation failed")
        except (MediaError, OSError) as exc:
            self.store.update(
                job_id,
                status="error",
                stage="failed",
                error=f"finalise failed: {exc}",
                finished=time.time(),
            )
            self.store.append_log(job_id, f"ERROR finalise: {exc}")

    # ------------------------------------------------------------------ render
    def _render(self, job: Job):
        engine = get_engine(job.engine)
        spec = self._build_spec(job)
        job.workdir.mkdir(parents=True, exist_ok=True)

        self.store.touch_progress(job.id, 0.005, "preparing sources")
        prepared = job.workdir / "prepared"
        try:
            spec.motion_path = prepare_motion(
                spec.motion_path,
                prepared / "motion.mp4",
                width=spec.width,
                height=spec.height,
                fps=spec.model_fps,
                frames=spec.frames,
                fit=str(job.params.get("fit") or "crop"),
            )
            spec.character_path = prepare_character(
                spec.character_path, prepared / "character.png"
            )
        except MediaError as exc:
            raise EngineError(f"could not prepare inputs: {exc}") from exc

        pinned = spec.motion_path
        if pinned.exists():
            info = probe(pinned)
            self.store.append_log(
                job.id,
                f"motion prepared: {info['width']}x{info['height']} "
                f"{info['nb_frames']}f @{info['fps']:.3f}fps",
            )

        def on_progress(value: float, stage: str) -> None:
            self.store.touch_progress(job.id, value, stage)

        def should_cancel() -> bool:
            return self.is_canceled(job.id)

        started = time.time()
        result = engine.run(spec, job.workdir, on_progress, should_cancel)
        self.store.append_log(
            job.id, f"engine finished in {round(time.time() - started, 1)}s"
        )
        if not result.path.exists():
            raise EngineError("engine reported success but produced no file")

        # Generate at Wan's native 16 fps, then optionally deliver 24/30/60 fps.
        # This preserves the requested duration while avoiding a 4x latent cost.
        target_fps = float(spec.fps)
        if abs(target_fps - float(spec.model_fps)) > 0.01:
            delivered = job.workdir / "render-delivered.mp4"
            retime_video(
                result.path,
                delivered,
                fps=target_fps,
                source_fps=float(spec.model_fps),
                duration=float(spec.duration),
            )
            result.path = delivered
            result.meta["fps_requested"] = target_fps
            result.meta["fps_model"] = float(spec.model_fps)
        return result

    def _build_spec(self, job: Job) -> JobSpec:
        params = job.params
        motion = job.motion_path
        character = job.character_path
        if not motion or not motion.exists():
            raise EngineError("motion source file is missing")
        if not character or not character.exists():
            raise EngineError("character reference file is missing")

        return JobSpec(
            job_id=job.id,
            motion_path=motion,
            character_path=character,
            prompt=str(params.get("prompt") or ""),
            negative=str(params.get("negative") or ""),
            width=int(params.get("width") or settings.default_width),
            height=int(params.get("height") or settings.default_height),
            fps=float(params.get("fps") or settings.default_fps),
            model_fps=float(params.get("model_fps") or min(16.0, float(params.get("fps") or settings.default_fps))),
            frames=int(params.get("frames") or 49),
            steps=int(params.get("steps") or 6),
            cfg=float(params.get("cfg") or 1.0),
            shift=float(params.get("shift") or 5.0),
            seed=int(params.get("seed") or 0),
            loop=bool(params.get("loop")),
            motion_info=job.source_info,
            character_info=job.character_info,
        )

    # ------------------------------------------------------------------- audio
    def _finalise_audio(self, job: Job, render: Path) -> tuple[Path, bool]:
        """Apply the "keep original audio" toggle. Returns (final_path, audio_used)."""
        out = job.output_dir / "result.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)

        keep = bool(job.params.get("keep_audio"))
        volume = float(job.params.get("audio_volume", 1.0) or 1.0)

        if not keep:
            strip_audio(render, out)
            self.store.append_log(job.id, "audio: stripped (toggle off)")
            return out, False

        if not job.audio_path:
            strip_audio(render, out)
            self.store.append_log(
                job.id, "audio: source had no audio track - rendered silently"
            )
            return out, False

        mux_audio(render, job.audio_path, out, volume=volume)
        self.store.append_log(
            job.id, f"audio: original track attached (gain {volume:.2f}x, trimmed/padded to video length)"
        )
        return out, True


runner: JobRunner | None = None


def get_runner(store: JobStore) -> JobRunner:
    global runner
    if runner is None:
        runner = JobRunner(store)
    return runner
