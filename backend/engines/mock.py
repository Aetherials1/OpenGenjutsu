"""Mock engine - renders a stand-in clip with ffmpeg only.

Purpose: lets the entire product (upload -> queue -> progress -> audio toggle ->
player -> download) be exercised on any laptop, with no GPU and no ComfyUI. The
output is intentionally obvious: the character image with the motion source
playing small in the corner, so nobody mistakes it for a real transfer.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..media import MediaError, ffmpeg_exe, probe, run
from .base import CancelFn, Engine, EngineError, EngineResult, JobSpec, ProgressFn


class MockEngine(Engine):
    id = "mock"
    label = "Mock (no GPU)"
    description = (
        "Renders the character image with the motion source inset. Exercises the "
        "whole pipeline without ComfyUI - use it to verify uploads, progress, the "
        "audio toggle and downloads."
    )
    maturity = "test"
    replacement_mode = "placeholder"
    notes = "Output is a placeholder composite, not a real motion transfer."

    def preflight(self) -> dict:
        from ..media import ffmpeg_status

        state = ffmpeg_status()
        problems = [] if state["ok"] else [state["error"] or "ffmpeg missing"]
        return {"ready": state["ok"], "problems": problems, "warnings": []}

    def defaults(self) -> dict:
        base = super().defaults()
        base.update({"steps": 1, "cfg": 1.0})
        return base

    def run(
        self,
        spec: JobSpec,
        workdir: Path,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> EngineResult:
        def report(value: float, stage: str) -> None:
            if progress:
                progress(max(0.0, min(0.99, value)), stage)

        def check_cancel() -> None:
            if cancel and cancel():
                raise EngineError("canceled")

        report(0.02, "reading sources")
        width = max(64, spec.width - spec.width % 16)
        height = max(64, spec.height - spec.height % 16)
        fps = spec.fps if spec.fps > 0 else 16.0
        duration = spec.duration or float(spec.motion_info.get("duration") or 4.0)
        duration = max(1.0, min(duration, 60.0))
        frames = max(1, int(round(duration * fps)))

        check_cancel()
        report(0.15, "compositing")

        out = workdir / "render.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)

        inset = max(48, int(width * 0.26))
        graph = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,fps={fps:g}[bg];"
            f"[1:v]scale={inset}:-2,setsar=1,fps={fps:g}[pip];"
            f"[bg][pip]overlay=W-w-24:H-h-24:shortest=0,format=yuv420p[v]"
        )

        cmd = [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-loop",
            "1",
            "-i",
            str(spec.character_path),
            "-stream_loop",
            "-1",
            "-i",
            str(spec.motion_path),
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-t",
            f"{duration:.3f}",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(out),
        ]

        try:
            run(cmd, timeout=900)
        except MediaError as exc:
            raise EngineError(f"mock render failed: {exc}") from exc

        # Fake a bit of sampling time so the progress UI is visible.
        steps = 8
        for i in range(steps):
            check_cancel()
            time.sleep(0.22)
            report(0.35 + 0.6 * (i + 1) / steps, "generating")

        info = probe(out)
        report(0.99, "finalising")
        return EngineResult(
            path=out,
            meta={
                "width": info["width"],
                "height": info["height"],
                "fps": round(info["fps"], 3),
                "duration": round(float(info["duration"]), 3),
                "frames": info["nb_frames"] or frames,
                "engine": self.id,
            },
        )
