#!/usr/bin/env python
"""End-to-end smoke test against a running OpenGenjutsu backend.

Verifies: health, upload -> queue -> render -> audio muxing (toggle ON and OFF)
-> file download. Run it after starting the backend:

    python scripts/smoke_test.py
    python scripts/smoke_test.py --url https://xxx.trycloudflare.com --token abc123

Exits non-zero on failure so it can gate a deploy/notebook cell.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f" - {detail}" if detail and not ok else ""))
        if not ok:
            self.failures += 1
        return ok


def wait_for_job(base: str, headers: dict, job_id: str, timeout: float = 900) -> dict:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        job = requests.get(f"{base}/api/jobs/{job_id}", headers=headers, timeout=60).json()
        line = f"{job['status']} {job['progress'] * 100:5.1f}% {job['stage']}"
        if line != last:
            print(f"      {line}")
            last = line
        if job["status"] in {"done", "error", "canceled"}:
            return job
        time.sleep(1.0)
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


def make_sources(tmp: Path) -> tuple[Path, Path]:
    """Build a small motion clip (with audio) and a character image with ffmpeg."""
    from backend import media

    motion = media.make_test_clip(tmp / "motion.mp4", seconds=3.0, width=480, height=320, fps=16)
    tone = media.make_test_tone(tmp / "tone.wav", seconds=3.0)
    with_audio = tmp / "motion_audio.mp4"
    media.mux_audio(motion, tone, with_audio, fade_out=0.2)
    character = media.make_test_image(tmp / "character.png", 480, 320)
    return with_audio, character


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--token", default="")
    parser.add_argument("--engine", default="")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    headers = {"X-Auth-Token": args.token} if args.token else {}
    check = Checker()

    print(f"\nOpenGenjutsu smoke test -> {base}\n")

    print("health")
    try:
        health = requests.get(f"{base}/api/health", headers=headers, timeout=30).json()
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] backend unreachable: {exc}")
        return 2

    check.check(health.get("ok") is True, "backend responds")
    check.check(bool(health.get("ffmpeg", {}).get("ok")), "ffmpeg available",
                str(health.get("ffmpeg", {}).get("error")))
    engines = {e["id"]: e for e in health.get("engines", [])}
    engine = args.engine or health.get("engine_default") or "mock"
    print(f"      engine under test: {engine}")
    check.check(engine in engines, "engine is registered", f"known: {sorted(engines)}")
    if engine in engines:
        check.check(engines[engine]["ready"], f"engine '{engine}' preflight",
                    "; ".join(engines[engine].get("problems") or []))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print("\nsources")
        motion, character = make_sources(tmp)
        info = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(motion)],
            stdout=subprocess.PIPE, text=True, check=False,
        )
        check.check("audio" in info.stdout, "test source has video+audio", info.stdout.strip())

        def submit(keep_audio: bool) -> dict:
            data = {
                "engine": engine,
                "width": "480",
                "height": "320",
                "fps": "16",
                "seconds": "3",
                "steps": "1" if engine == "mock" else "4",
                "cfg": "1",
                "seed": "7",
                "keep_audio": "true" if keep_audio else "false",
                "audio_volume": "1",
                "prompt": "smoke test",
            }
            with open(motion, "rb") as m, open(character, "rb") as c:
                response = requests.post(
                    f"{base}/api/jobs",
                    headers=headers,
                    files={"motion": ("motion.mp4", m, "video/mp4"),
                           "character": ("character.png", c, "image/png")},
                    data=data,
                    timeout=600,
                )
            if response.status_code != 201:
                raise RuntimeError(f"upload failed {response.status_code}: {response.text[:400]}")
            return response.json()

        print("\nrender with original audio ON")
        job = submit(True)
        done = wait_for_job(base, headers, job["id"], args.timeout)
        check.check(done["status"] == "done", "job finished", done.get("error") or "")
        check.check(done.get("has_source_audio") is True, "source audio detected")
        check.check(done.get("audio_applied") is True, "audio muxed")
        check.check(bool(done["meta"].get("has_audio")), "result contains an audio stream")

        result = requests.get(f"{base}/api/jobs/{job['id']}/result", headers=headers, timeout=300)
        check.check(result.status_code == 200 and len(result.content) > 5000, "result downloadable",
                    f"{result.status_code}, {len(result.content)} bytes")
        out_with = tmp / "with_audio.mp4"
        out_with.write_bytes(result.content)
        check.check(abs(float(done["meta"].get("duration", 0)) - 3.06) < 1.2,
                    "duration matches request", f"{done['meta'].get('duration')}s")

        print("\nrender with original audio OFF")
        job2 = submit(False)
        done2 = wait_for_job(base, headers, job2["id"], args.timeout)
        check.check(done2["status"] == "done", "job finished", done2.get("error") or "")
        check.check(done2.get("audio_applied") is False, "audio not applied")
        check.check(not bool(done2["meta"].get("has_audio")), "result has no audio stream")

        print("\ndeliverables")
        for path in (out_with,):
            check.check(path.stat().st_size > 5000, f"{path.name} written ({path.stat().st_size} bytes)")

    print()
    if check.failures:
        print(f"{check.failures} check(s) FAILED\n")
        return 1
    print("all checks passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
