"""ffmpeg / ffprobe helpers: probing, audio extraction, audio muxing, thumbnails.

Everything media-related lives here so the audio toggle is implemented in exactly
one place (see ``mux_audio``).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

FFMPEG_FALLBACK_HINT = (
    "ffmpeg was not found. Install it (Windows: `winget install Gyan.FFmpeg`, "
    "Debian/Colab: `apt-get install -y ffmpeg`) or set OG_FFMPEG / OG_FFPROBE."
)

_ffmpeg: str | None = None
_ffprobe: str | None = None
_ffmpeg_resolved = False


class MediaError(RuntimeError):
    """Raised when a media operation fails."""


# --------------------------------------------------------------------------- #
# binary discovery
# --------------------------------------------------------------------------- #


def _existing(path: str | None) -> str | None:
    if path and Path(path).expanduser().exists():
        return str(Path(path).expanduser())
    return None


def ffmpeg_exe() -> str:
    global _ffmpeg, _ffmpeg_resolved
    if _ffmpeg:
        return _ffmpeg
    _ffmpeg_resolved = True

    candidate = _existing(os.getenv("OG_FFMPEG")) or shutil.which("ffmpeg")
    if candidate:
        _ffmpeg = candidate
        return candidate

    # Last resort: the binary bundled with the imageio-ffmpeg wheel.
    try:
        import imageio_ffmpeg  # type: ignore

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            _ffmpeg = bundled
            return bundled
    except Exception:  # pragma: no cover - optional dependency
        pass

    raise MediaError(FFMPEG_FALLBACK_HINT)


def ffprobe_exe() -> str | None:
    global _ffprobe
    if _ffprobe:
        return _ffprobe

    candidate = _existing(os.getenv("OG_FFPROBE")) or shutil.which("ffprobe")
    if candidate:
        _ffprobe = candidate
        return candidate

    # Look next to the ffmpeg we found (common for portable installs).
    try:
        sibling = Path(ffmpeg_exe()).with_name(
            "ffprobe.exe" if os.name == "nt" else "ffprobe"
        )
        if sibling.exists():
            _ffprobe = str(sibling)
            return _ffprobe
    except MediaError:
        return None
    return None


def ffmpeg_status() -> dict[str, Any]:
    info: dict[str, Any] = {"ffmpeg": None, "ffprobe": None, "ok": False, "error": None}
    try:
        info["ffmpeg"] = ffmpeg_exe()
        info["ffprobe"] = ffprobe_exe()
        info["ok"] = True
        if not info["ffprobe"]:
            info["error"] = (
                "ffprobe missing - video metadata probing will use a slower fallback."
            )
    except MediaError as exc:
        info["error"] = str(exc)
    return info


# --------------------------------------------------------------------------- #
# process helpers
# --------------------------------------------------------------------------- #


def run(cmd: list[str], timeout: int = 900) -> str:
    """Run a subprocess, returning stdout. Raises MediaError on failure."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"{cmd[0]} not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"command timed out after {timeout}s: {cmd[0]}") from exc

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        tail = "\n".join(stderr.splitlines()[-12:])
        raise MediaError(f"{Path(cmd[0]).name} failed ({proc.returncode}):\n{tail}")
    return proc.stdout.decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# probing
# --------------------------------------------------------------------------- #


def _ratio(value: str | None) -> float:
    if not value or value in {"0/0", "N/A", "NA"}:
        return 0.0
    try:
        if "/" in value:
            num, _, den = value.partition("/")
            n, d = float(num), float(den)
            result = n / d if d else 0.0
        else:
            result = float(value)
        return result if math.isfinite(result) and result >= 0 else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        result = float(value)
        return int(result) if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _empty_probe() -> dict[str, Any]:
    return {
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "nb_frames": 0,
        "has_video": False,
        "has_audio": False,
        "audio_channels": 0,
        "video_codec": "",
        "audio_codec": "",
        "size_bytes": 0,
        "format_name": "",
    }


def _probe_json(ffprobe: str, path: Path) -> dict[str, Any]:
    raw = run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=120,
    )
    data = json.loads(raw or "{}")
    info = _empty_probe()
    fmt = data.get("format") or {}
    info["format_name"] = fmt.get("format_name", "")
    try:
        info["size_bytes"] = _safe_int(fmt.get("size"), path.stat().st_size)
    except OSError:
        info["size_bytes"] = 0
    info["duration"] = max(0.0, _safe_float(fmt.get("duration")))

    for stream in data.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video" and not info["has_video"]:
            # Skip embedded cover art.
            if (stream.get("disposition") or {}).get("attached_pic"):
                continue
            info["has_video"] = True
            info["width"] = max(0, _safe_int(stream.get("width")))
            info["height"] = max(0, _safe_int(stream.get("height")))
            info["video_codec"] = stream.get("codec_name", "")
            info["fps"] = _ratio(stream.get("avg_frame_rate")) or _ratio(
                stream.get("r_frame_rate")
            )
            info["nb_frames"] = max(0, _safe_int(stream.get("nb_frames")))
            if not info["duration"]:
                info["duration"] = max(0.0, _safe_float(stream.get("duration")))
        elif kind == "audio" and not info["has_audio"]:
            info["has_audio"] = True
            info["audio_codec"] = stream.get("codec_name", "")
            info["audio_channels"] = max(0, _safe_int(stream.get("channels")))

    if not info["nb_frames"] and info["duration"] and info["fps"]:
        info["nb_frames"] = int(round(info["duration"] * info["fps"]))
    return info


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_RES_RE = re.compile(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")


def _probe_via_ffmpeg(path: Path) -> dict[str, Any]:
    """Fallback for machines without ffprobe: parse `ffmpeg -i` stderr."""
    ffmpeg = ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    text = proc.stderr.decode("utf-8", "replace")
    if "Invalid data found" in text or "No such file" in text:
        raise MediaError(f"cannot read media file: {path.name}")

    info = _empty_probe()
    info["size_bytes"] = path.stat().st_size
    match = _DUR_RE.search(text)
    if match:
        hours, minutes, seconds = match.groups()
        info["duration"] = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if re.search(r"Stream #\S+.*: Video:", text):
        info["has_video"] = True
        res = _RES_RE.search(text)
        if res:
            info["width"], info["height"] = int(res.group(1)), int(res.group(2))
        fps = _FPS_RE.search(text)
        if fps:
            info["fps"] = float(fps.group(1))
    if re.search(r"Stream #\S+.*: Audio:", text):
        info["has_audio"] = True
    if not info["has_video"] and not info["has_audio"]:
        raise MediaError(f"no audio or video stream found in {path.name}")
    return info


def probe(path: str | Path) -> dict[str, Any]:
    """Return metadata for a media file. Never raises for missing streams."""
    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        raise MediaError(f"empty or missing file: {target.name}")

    ffprobe = ffprobe_exe()
    if ffprobe:
        try:
            return _probe_json(ffprobe, target)
        except (MediaError, json.JSONDecodeError):
            pass
    return _probe_via_ffmpeg(target)


def has_audio(path: str | Path) -> bool:
    try:
        return bool(probe(path)["has_audio"])
    except MediaError:
        return False


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #


def extract_audio(source: str | Path, dest: str | Path) -> Path:
    """Extract the first audio stream to a 48kHz stereo WAV."""
    src, out = Path(source), Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(
            [
                ffmpeg_exe(),
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-vn",
                "-map",
                "0:a:0",
                "-ac",
                "2",
                "-ar",
                "48000",
                "-c:a",
                "pcm_s16le",
                str(out),
            ]
        )
        if not out.exists() or out.stat().st_size == 0:
            raise MediaError("audio extraction produced an empty file")
    except (MediaError, OSError):
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return out


def mux_audio(
    video: str | Path,
    audio_source: str | Path,
    dest: str | Path,
    *,
    volume: float = 1.0,
    fade_out: float = 0.35,
    pad_to_video: bool = True,
) -> Path:
    """Attach ``audio_source``'s audio to ``video`` and write ``dest``.

    Duration is always matched to the *video*: the audio is padded with silence
    when the render is longer, and cut when the render is shorter. `-shortest`
    plus `apad` does exactly that, so the output never ends with silence or a
    hard mid-word cut.

    ``volume`` is a linear gain (1.0 = untouched, 0.0 = silent -> audio is then
    omitted entirely so the file stays clean).
    """
    src_video, src_audio, out = Path(video), Path(audio_source), Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)

    if volume <= 0.001:
        # "keep audio" off / muted: just copy the video stream through.
        run(
            [
                ffmpeg_exe(),
                "-y",
                "-v",
                "error",
                "-i",
                str(src_video),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-an",
                str(out),
            ]
        )
        return out

    info = probe(src_video)
    duration = float(info["duration"] or 0.0)

    filters: list[str] = []
    if abs(volume - 1.0) > 0.001:
        filters.append(f"volume={max(0.0, min(8.0, volume)):.3f}")
    if fade_out > 0 and duration > fade_out * 2:
        start = max(0.0, duration - fade_out)
        filters.append(f"afade=t=out:st={start:.3f}:d={fade_out:.3f}")
    if pad_to_video:
        filters.append("apad")
    chain = ",".join(filters) if filters else "anull"

    run(
        [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-i",
            str(src_video),
            "-i",
            str(src_audio),
            "-filter_complex",
            f"[1:a:0]{chain}[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out),
        ],
        timeout=600,
    )
    return out


def strip_audio(video: str | Path, dest: str | Path) -> Path:
    out = Path(dest)
    run(
        [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            str(out),
        ]
    )
    return out


# --------------------------------------------------------------------------- #
# video normalisation / helpers
# --------------------------------------------------------------------------- #


def retime_video(
    source: str | Path,
    dest: str | Path,
    *,
    fps: float,
    source_fps: float = 0.0,
    duration: float | None = None,
) -> Path:
    """Deliver a render at ``fps`` without changing its duration.

    Wan Animate samples at 16 fps.  For higher delivery rates, motion
    interpolation creates in-between frames instead of merely duplicating
    frames; the diffusion model still only has to generate its native frames.
    """
    src, out = Path(source), Path(dest)
    target = max(1.0, float(fps))
    source_rate = float(source_fps or 0.0)
    if not source_rate:
        source_rate = float(probe(src).get("fps") or 0.0)

    if source_rate and target > source_rate + 0.01:
        vf = (
            f"minterpolate=fps={target:g}:mi_mode=mci:mc_mode=aobmc:"
            "me_mode=bidir:vsbmc=1"
        )
    else:
        vf = f"fps={target:g}"

    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_exe(), "-y", "-v", "error", "-i", str(src),
        "-vf", vf,
    ]
    if duration and duration > 0:
        command += ["-t", f"{float(duration):g}"]
    command += [
        "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-an",
        "-movflags", "+faststart", str(out),
    ]
    run(command, timeout=1800)
    if not out.exists() or out.stat().st_size == 0:
        raise MediaError("video retiming produced an empty file")
    return out


def ensure_mp4(source: str | Path, dest: str | Path, fps: float | None = None) -> Path:
    """Convert anything ffmpeg can read (e.g. animated webp) into a clean H.264 mp4."""
    src, out = Path(source), Path(dest)
    if src.suffix.lower() == ".mp4" and src.resolve() != out.resolve():
        shutil.copy2(src, out)
        return out

    info = probe(src)
    vf = ["scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-vf",
        ",".join(vf),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
    ]
    if fps and fps > 0:
        cmd += ["-r", f"{fps:g}"]
    elif info["fps"] > 0:
        cmd += ["-r", f"{info['fps']:.3f}"]
    cmd.append(str(out))
    run(cmd, timeout=900)
    return out


def thumbnail(video: str | Path, dest: str | Path, at: float = 0.6) -> Path | None:
    out = Path(dest)
    try:
        run(
            [
                ffmpeg_exe(),
                "-y",
                "-v",
                "error",
                "-ss",
                f"{max(0.0, at):.2f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=480:-2",
                str(out),
            ],
            timeout=120,
        )
        return out if out.exists() and out.stat().st_size else None
    except MediaError:
        return None


def make_test_clip(
    dest: str | Path,
    *,
    seconds: float = 4.0,
    width: int = 832,
    height: int = 480,
    fps: float = 16.0,
    text: str = "MOTION SOURCE",
) -> Path:
    """Synthesise a clip with obvious frame-to-frame motion (pipeline self-test)."""
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1.0, float(seconds))
    width = max(64, width - width % 2)
    height = max(64, height - height % 2)
    # testsrc2 animates on its own, so this always yields real motion.
    graph = (
        f"testsrc2=size={width}x{height}:rate={fps:g}:duration={duration},"
        f"eq=saturation=0.55:brightness=-0.06,"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x36e0c8@0.9:t={max(2, height // 140)},"
        f"format=yuv420p"
    )
    run(
        [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            graph,
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
            str(out),
        ],
        timeout=300,
    )
    del text
    return out


def make_test_image(dest: str | Path, width: int = 832, height: int = 480) -> Path:
    """Synthesise a still "character" image (a stylised silhouette) as PNG."""
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = (
        f"color=c=0x14121c:s={width}x{height},"
        f"drawbox=x={width // 3}:y={height // 2}:w={width // 3}:h={height // 2}:"
        f"color=0x6f5cff@1.0:t=fill,"
        f"drawbox=x={width // 2 - width // 12}:y={height // 4}:"
        f"w={width // 6}:h={height // 4}:color=0xe8b98f@1.0:t=fill"
    )
    run(
        [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            graph,
            "-frames:v",
            "1",
            "-update",
            "1",
            str(out),
        ],
        timeout=60,
    )
    return out


def make_test_tone(dest: str | Path, seconds: float = 4.0) -> Path:
    """Synthesise a short WAV with two notes, used to verify the audio toggle."""
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={max(0.5, seconds)}",
            "-c:a",
            "pcm_s16le",
            str(out),
        ],
        timeout=120,
    )
    return out


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


# --------------------------------------------------------------------------- #
# input normalisation (run before every generation)
# --------------------------------------------------------------------------- #


def prepare_motion(
    source: str | Path,
    dest: str | Path,
    *,
    width: int,
    height: int,
    fps: float,
    frames: int,
    fit: str = "crop",
) -> Path:
    """Re-time the motion source to exactly ``frames`` @ ``fps`` at ``width x height``.

    Doing this before inference means every engine receives a clip whose length
    matches the requested latent length, so no engine has to guess, pad or fail
    on a mismatched control video. Short sources are looped, long ones trimmed.
    """
    src, out = Path(source), Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1.0, float(fps))
    frames = max(5, int(frames))
    width = max(64, int(width) - int(width) % 2)
    height = max(64, int(height) - int(height) % 2)

    if fit == "stretch":
        scale = f"scale={width}:{height}"
    elif fit == "pad":
        scale = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    else:  # crop / cover
        scale = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )

    vf = f"fps={fps:g},{scale},setsar=1,format=yuv420p"
    run(
        [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-stream_loop",
            "-1",
            "-i",
            str(src),
            "-vf",
            vf,
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
        timeout=900,
    )
    info = probe(out)
    if info["nb_frames"] < frames - 2:
        raise MediaError(
            f"could not build a {frames}-frame motion clip from {src.name} "
            f"(got {info['nb_frames']})"
        )
    return out


def prepare_character(source: str | Path, dest: str | Path, max_side: int = 1280) -> Path:
    """Normalise the reference image to a plain RGB PNG that every loader accepts."""
    src, out = Path(source), Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-vf",
            f"scale='min({max_side},iw)':'min({max_side},ih)':"
            "force_original_aspect_ratio=decrease,format=rgb24",
            "-frames:v",
            "1",
            "-update",
            "1",
            str(out),
        ],
        timeout=300,
    )
    if not out.exists() or out.stat().st_size == 0:
        raise MediaError(f"could not read character image {src.name}")
    return out
