"""Job persistence: one directory per job, one JSON file of state.

Layout::

    data/jobs/<job_id>/
        job.json            # state + parameters
        job.log             # full log
        input/motion.mp4    # uploaded motion source
        input/character.png # uploaded character reference
        input/audio.wav     # audio extracted from the motion source
        output/result.mp4   # final render (with or without audio)
        output/thumb.jpg
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import settings

TERMINAL = {"done", "error", "canceled"}
MAX_LOG_LINES = 400


def _now() -> float:
    return time.time()


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error | canceled
    progress: float = 0.0
    stage: str = "queued"
    engine: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created: float = field(default_factory=_now)
    started: float | None = None
    finished: float | None = None
    source_info: dict[str, Any] = field(default_factory=dict)
    character_info: dict[str, Any] = field(default_factory=dict)
    has_source_audio: bool = False
    audio_applied: bool = False

    # ---- derived ----
    @property
    def dir(self) -> Path:
        return settings.jobs_dir / self.id

    @property
    def input_dir(self) -> Path:
        return self.dir / "input"

    @property
    def output_dir(self) -> Path:
        return self.dir / "output"

    @property
    def workdir(self) -> Path:
        return self.dir / "work"

    @property
    def motion_path(self) -> Path | None:
        name = self.params.get("motion_file")
        return self.input_dir / name if name else None

    @property
    def character_path(self) -> Path | None:
        name = self.params.get("character_file")
        return self.input_dir / name if name else None

    @property
    def audio_path(self) -> Path | None:
        path = self.input_dir / "audio.wav"
        return path if path.exists() and path.stat().st_size else None

    @property
    def result_path(self) -> Path | None:
        path = self.output_dir / "result.mp4"
        return path if path.exists() else None

    @property
    def thumb_path(self) -> Path | None:
        path = self.output_dir / "thumb.jpg"
        return path if path.exists() else None

    @property
    def elapsed(self) -> float | None:
        if self.started is None:
            return None
        return (self.finished or _now()) - self.started

    def to_api(self, queue_position: int | None = None) -> dict[str, Any]:
        base = f"/api/jobs/{self.id}"
        return {
            "id": self.id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "engine": self.engine,
            "params": self.params,
            "meta": self.meta,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "elapsed": round(self.elapsed, 2) if self.elapsed else None,
            "source_info": self.source_info,
            "character_info": self.character_info,
            "has_source_audio": self.has_source_audio,
            "audio_applied": self.audio_applied,
            "queue_position": queue_position,
            "urls": {
                "result": f"{base}/result" if self.result_path else None,
                "source": f"{base}/source" if self.motion_path else None,
                "character": f"{base}/character" if self.character_path else None,
                "thumb": f"{base}/thumb" if self.thumb_path else None,
            },
            "logs": logs.get(self.id, [])[-60:],
        }

    def to_disk(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "engine": self.engine,
            "params": self.params,
            "meta": self.meta,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "source_info": self.source_info,
            "character_info": self.character_info,
            "has_source_audio": self.has_source_audio,
            "audio_applied": self.audio_applied,
        }
        return data

    @classmethod
    def from_disk(cls, data: dict[str, Any]) -> "Job":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)


# job_id -> recent log lines (memory only; full log lives on disk)
logs: dict[str, list[str]] = {}


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._dirty_times: dict[str, float] = {}

    # --------------------------------------------------------------- lifecycle
    def load(self) -> None:
        settings.ensure_dirs()
        for directory in sorted(settings.jobs_dir.glob("*")):
            state = directory / "job.json"
            if not state.is_file():
                continue
            try:
                data = json.loads(state.read_text(encoding="utf-8"))
                job = Job.from_disk(data)
            except (OSError, ValueError, TypeError):
                continue
            # Anything "running" when the process died is gone for good.
            if job.status in {"queued", "running"}:
                job.status = "error"
                job.stage = "interrupted"
                job.error = "backend restarted before this job finished"
                job.finished = job.finished or _now()
                self._write(job)
            self._jobs[job.id] = job
            self._load_log(job.id)

    def _load_log(self, job_id: str) -> None:
        path = settings.jobs_dir / job_id / "job.log"
        if not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        logs[job_id] = lines[-MAX_LOG_LINES:]

    def create(self, params: dict[str, Any], engine: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], params=dict(params), engine=engine)
        job.input_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job.id] = job
            logs[job.id] = []
            self._write(job)
        return job

    # ------------------------------------------------------------------ access
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def queued_ids(self) -> list[str]:
        with self._lock:
            queued = [j for j in self._jobs.values() if j.status == "queued"]
        return [j.id for j in sorted(queued, key=lambda j: j.created)]

    def queue_position(self, job_id: str) -> int | None:
        queued = self.queued_ids()
        if job_id in queued:
            return queued.index(job_id) + 1
        return None

    def api_list(self, limit: int = 30) -> list[dict[str, Any]]:
        jobs = self.all()[:limit]
        with self._lock:
            return [job.to_api(self.queue_position(job.id)) for job in jobs]

    def api_one(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_api(self.queue_position(job_id)) if job else None

    # ----------------------------------------------------------------- updates
    def update(self, job_id: str, *, log: str | None = None, **fields: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            if log:
                self._append_log(job_id, log)
            self._write(job)
            return job

    def touch_progress(self, job_id: str, progress: float, stage: str) -> None:
        """Cheap progress update, throttled disk writes."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            changed_stage = job.stage != stage
            job.progress = max(job.progress, progress)
            job.stage = stage
            now = _now()
            last = self._dirty_times.get(job_id, 0.0)
            if changed_stage or now - last > 1.0 or progress >= 0.99:
                self._dirty_times[job_id] = now
                self._write(job)

    def append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            self._append_log(job_id, line)

    def _append_log(self, job_id: str, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        entry = f"[{stamp}] {line}"
        bucket = logs.setdefault(job_id, [])
        bucket.append(entry)
        if len(bucket) > MAX_LOG_LINES:
            del bucket[: len(bucket) - MAX_LOG_LINES]
        try:
            path = settings.jobs_dir / job_id / "job.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(entry + "\n")
        except OSError:
            pass

    def _write(self, job: Job) -> None:
        try:
            job.dir.mkdir(parents=True, exist_ok=True)
            tmp = job.dir / "job.json.tmp"
            tmp.write_text(json.dumps(job.to_disk(), indent=2), encoding="utf-8")
            tmp.replace(job.dir / "job.json")
        except OSError:
            pass

    # ----------------------------------------------------------------- cleanup
    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            logs.pop(job_id, None)
        if not job:
            return False
        shutil.rmtree(job.dir, ignore_errors=True)
        return True

    def prune(self, keep: int | None = None) -> int:
        keep = settings.keep_jobs if keep is None else keep
        removed = 0
        with self._lock:
            finished = [j for j in self._jobs.values() if j.status in TERMINAL]
        finished.sort(key=lambda j: j.created, reverse=True)
        for job in finished[keep:]:
            if self.delete(job.id):
                removed += 1
        return removed

    def purge_workdirs(self) -> None:
        for job in self.all():
            if job.status in TERMINAL:
                shutil.rmtree(job.workdir, ignore_errors=True)


store = JobStore()
