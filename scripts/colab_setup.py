#!/usr/bin/env python
"""One-shot bootstrap for running OpenGenjutsu with ComfyUI on a Colab GPU.

Subcommands (each is safe to re-run):

    system      install apt packages (ffmpeg, aria2)
    comfy       clone/update ComfyUI and install its python deps
    models      fetch the model weights (see download_models.py)
    serve       start ComfyUI + the OpenGenjutsu backend, wait until healthy
    tunnel      open a public cloudflared URL for the backend
    status      show process + health state
    doctor      print engine readiness diagnostics
    stop        stop everything started by `serve`

Typical Colab sequence::

    !python scripts/colab_setup.py system
    !python scripts/colab_setup.py comfy
    !python scripts/colab_setup.py models --preset vace
    !python scripts/colab_setup.py serve
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path(os.getenv("OG_LOG_DIR", "/content/ogenjutsu-logs" if Path("/content").exists() else PROJECT_ROOT / "logs"))
PID_FILE = LOG_DIR / "pids.json"
COMFY_DIR = Path(os.getenv("OG_COMFY_DIR", "/content/ComfyUI" if Path("/content").exists() else PROJECT_ROOT.parent / "ComfyUI"))
COMFY_PORT = int(os.getenv("OG_COMFY_PORT", "8188"))
BACKEND_PORT = int(os.getenv("OG_PORT", "7860"))
COMFY_URL = f"http://127.0.0.1:{COMFY_PORT}"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def log(message: str) -> None:
    print(f"[ogenjutsu] {message}", flush=True)


def step(message: str) -> None:
    print(f"\n=== {message} " + "=" * max(0, 58 - len(message)), flush=True)


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True, env: dict | None = None) -> int:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def pip(*args: str) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", *args])


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        import requests

        return requests.get(url, timeout=timeout).status_code < 500
    except Exception:
        return False


def wait_for(url: str, label: str, timeout: float = 240.0) -> bool:
    started = time.time()
    while time.time() - started < timeout:
        if http_ok(url):
            log(f"{label} is up ({round(time.time() - started, 1)}s)")
            return True
        time.sleep(2)
    log(f"{label} did not respond after {timeout:.0f}s")
    return False


def vram_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        return int(float(out.stdout.decode().strip().splitlines()[0]))
    except Exception:
        return 0


def load_pids() -> dict:
    if PID_FILE.exists():
        try:
            return json.loads(PID_FILE.read_text())
        except ValueError:
            return {}
    return {}


def save_pids(data: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(json.dumps(data, indent=2))


def kill_pid(pid: int, label: str) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
        log(f"stopped {label} ({pid})")
    except ProcessLookupError:
        log(f"{label} ({pid}) already gone")
    except Exception as exc:  # noqa: BLE001
        log(f"could not stop {label} ({pid}): {exc}")


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def cmd_system(args: argparse.Namespace) -> int:
    step("system packages")
    missing = [b for b in ("ffmpeg", "ffprobe", "git") if not shutil.which(b)]
    if missing or args.force:
        log(f"installing: {', '.join(missing) or 'all (forced)'}")
        run(["apt-get", "update", "-qq"], check=False)
        run(["apt-get", "install", "-y", "-qq", "ffmpeg", "git", "curl"], check=False)
    else:
        log("ffmpeg / git already present")

    # aria2 is optional (faster HF downloads) - never fatal.
    if not shutil.which("aria2c"):
        run(["apt-get", "install", "-y", "-qq", "aria2"], check=False)

    log(f"ffmpeg -> {shutil.which('ffmpeg') or 'MISSING'}")
    return 0


def cmd_comfy(args: argparse.Namespace) -> int:
    step("ComfyUI")
    if not (COMFY_DIR / "main.py").is_file():
        run(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_DIR)])
    else:
        log(f"updating existing checkout at {COMFY_DIR}")
        run(["git", "pull", "--ff-only"], cwd=COMFY_DIR, check=False)

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=COMFY_DIR)

    if args.with_gguf:
        custom = COMFY_DIR / "custom_nodes"
        custom.mkdir(exist_ok=True)
        target = custom / "ComfyUI-GGUF"
        if not target.is_dir():
            run(["git", "clone", "--depth", "1", "https://github.com/city96/ComfyUI-GGUF.git", str(target)])
            run([sys.executable, "-m", "pip", "install", "-q", "-r", str(target / "requirements.txt")], check=False)
        else:
            log("ComfyUI-GGUF already installed")

    log(f"ComfyUI ready at {COMFY_DIR}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    step("model weights")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "download_models.py"),
        "--preset",
        args.preset,
        "--comfy-root",
        str(COMFY_DIR),
    ]
    if args.models_root:
        cmd += ["--models-root", args.models_root]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.quant:
        cmd += ["--quant", args.quant]
    return run(cmd, check=False)


def comfy_flags(profile: str) -> list[str]:
    vram = vram_mb()
    if profile == "auto":
        profile = "big" if vram >= 18000 else "t4"
    log(f"GPU VRAM: {vram or '?'} MB -> profile '{profile}'")

    if profile == "big":
        return ["--preview-method", "none"]
    return ["--lowvram", "--preview-method", "none", "--use-split-cross-attention"]


def cmd_serve(args: argparse.Namespace) -> int:
    step("starting ComfyUI")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pids = load_pids()

    if not (COMFY_DIR / "main.py").is_file():
        log("ComfyUI is not installed - run: python scripts/colab_setup.py comfy")
        return 1

    listen = "0.0.0.0" if args.expose_comfy else "127.0.0.1"
    comfy_log = open(LOG_DIR / "comfyui.log", "w", encoding="utf-8", buffering=1)
    comfy = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "--listen",
            listen,
            "--port",
            str(COMFY_PORT),
            *comfy_flags(args.profile),
        ],
        cwd=str(COMFY_DIR),
        stdout=comfy_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pids["comfy"] = comfy.pid
    save_pids(pids)
    log(f"ComfyUI pid {comfy.pid} -> {LOG_DIR / 'comfyui.log'}")

    if not wait_for(f"{COMFY_URL}/system_stats", "ComfyUI", timeout=args.timeout):
        log("check the log above; ComfyUI usually fails here on missing torch/CUDA")
        return 1

    step("starting OpenGenjutsu backend")
    env = os.environ.copy()
    env.update(
        {
            "OG_COMFY_URL": COMFY_URL,
            "OG_COMFY_INPUT_DIR": str(COMFY_DIR / "input"),
            "OG_DATA_DIR": args.data_dir or str(PROJECT_ROOT / "data"),
            "OG_TOKEN": args.token or os.getenv("OG_TOKEN", ""),
            "OG_ENGINE": args.engine,
            "OG_CORS": "*",
            "OG_PORT": str(BACKEND_PORT),
            "PYTHONUNBUFFERED": "1",
        }
    )
    backend_log = open(LOG_DIR / "backend.log", "w", encoding="utf-8", buffering=1)
    backend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(BACKEND_PORT),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=backend_log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    pids["backend"] = backend.pid
    if env["OG_TOKEN"]:
        pids["token"] = env["OG_TOKEN"]
    save_pids(pids)
    log(f"backend pid {backend.pid} -> {LOG_DIR / 'backend.log'}")

    if not wait_for(f"http://127.0.0.1:{BACKEND_PORT}/api/health", "backend", timeout=180):
        return 1

    cmd_status(args)
    log("")
    log("Next: open a tunnel so your browser can reach this runtime.")
    log("    !python scripts/colab_setup.py tunnel")
    return 0


def cmd_tunnel(args: argparse.Namespace) -> int:
    step("cloudflared tunnel")
    binary = Path(args.binary) if args.binary else Path("/usr/local/bin/cloudflared")

    if not binary.exists():
        log("downloading cloudflared…")
        url = (
            "https://github.com/cloudflare/cloudflared/releases/latest/download/"
            + ("cloudflared-windows-amd64.exe" if os.name == "nt" else "cloudflared-linux-amd64")
        )
        run(["curl", "-L", "-o", str(binary), url])
        if os.name != "nt":
            binary.chmod(0o755)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tunnel_log = LOG_DIR / "tunnel.log"
    handle = open(tunnel_log, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        [str(binary), "tunnel", "--url", f"http://127.0.0.1:{BACKEND_PORT}", "--no-autoupdate"],
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pids = load_pids()
    pids["tunnel"] = proc.pid
    save_pids(pids)

    log("waiting for the public URL…")
    public = None
    started = time.time()
    while time.time() - started < 90:
        text = tunnel_log.read_text(encoding="utf-8", errors="replace")
        for token in text.replace("\r", " ").split():
            if token.startswith("https://") and "trycloudflare.com" in token:
                public = token.strip()
                break
        if public:
            break
        time.sleep(1)

    if not public:
        log(f"no URL found yet - check {tunnel_log} (the log keeps updating)")
        return 1

    token = os.getenv("OG_TOKEN", "") or (load_pids().get("token") or "")
    step("READY")
    print(f"\n  Open this URL:  {public}")
    if token:
        print(f"  Access token:   {token}")
    print("\n  Keep this cell's runtime alive. Ctrl+C / `stop` ends the session.\n")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    pids = load_pids()
    step("status")
    for name, pid in pids.items():
        if name == "token":
            continue
        alive = False
        try:
            os.kill(int(pid), 0)
            alive = True
        except Exception:
            pass
        log(f"{name:8s} pid {pid} {'running' if alive else 'stopped'}")

    log(f"ComfyUI   {COMFY_URL} {'ok' if http_ok(f'{COMFY_URL}/system_stats') else 'down'}")
    health_url = f"http://127.0.0.1:{BACKEND_PORT}/api/health"
    log(f"backend   {health_url} {'ok' if http_ok(health_url) else 'down'}")
    log(f"logs      {LOG_DIR}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    step("doctor")
    try:
        import requests
    except ImportError:
        log("requests missing - pip install requests")
        return 1

    token = os.getenv("OG_TOKEN", "")
    headers = {"X-Auth-Token": token} if token else {}
    try:
        health = requests.get(
            f"http://127.0.0.1:{BACKEND_PORT}/api/health", headers=headers, timeout=20
        ).json()
    except Exception as exc:  # noqa: BLE001
        log(f"backend not reachable: {exc}")
        return 1

    log(f"version        {health.get('version')}")
    log(f"comfyui        {health.get('comfy', {}).get('reachable')} ({health.get('comfy', {}).get('url')})")
    gpu = health.get("gpu") or {}
    log(f"gpu            {gpu.get('name')} {gpu.get('vram_total_mb')} MB" if gpu.get("available") else "gpu            none")
    log(f"ffmpeg         {health.get('ffmpeg', {}).get('ok')} {health.get('ffmpeg', {}).get('error') or ''}")
    log(f"engine default {health.get('engine_default')}")
    for engine in health.get("engines") or []:
        mark = "READY  " if engine.get("ready") else "BLOCKED"
        log(f"  [{mark}] {engine['id']} - {engine.get('label')}")
        for problem in engine.get("problems") or []:
            log(f"             ! {problem}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    pids = load_pids()
    for name in ("tunnel", "backend", "comfy"):
        pid = pids.get(name)
        if pid:
            kill_pid(int(pid), name)
    if PID_FILE.exists():
        PID_FILE.unlink()
    return 0


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenGenjutsu Colab bootstrap")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("system", help="install apt packages")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_system)

    p = sub.add_parser("comfy", help="install/update ComfyUI")
    p.add_argument("--with-gguf", action="store_true", help="also install the GGUF loader node")
    p.set_defaults(func=cmd_comfy)

    p = sub.add_parser("models", help="download model weights")
    p.add_argument("--preset", default="vace", choices=["vace", "animate", "animate-q4", "all"])
    p.add_argument("--quant", default="", choices=["", "fp8", "q4"])
    p.add_argument("--models-root", default=os.getenv("OG_MODELS_DIR", ""))
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("serve", help="start ComfyUI + backend")
    p.add_argument("--profile", default="auto", choices=["auto", "t4", "big"])
    p.add_argument("--engine", default=os.getenv("OG_ENGINE", "auto"))
    p.add_argument("--token", default=os.getenv("OG_TOKEN", ""))
    p.add_argument("--data-dir", default=os.getenv("OG_DATA_DIR", ""))
    p.add_argument("--timeout", type=float, default=240)
    p.add_argument("--expose-comfy", action="store_true", help="also listen on 0.0.0.0:8188")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("tunnel", help="open a public URL")
    p.add_argument("--binary", default="")
    p.set_defaults(func=cmd_tunnel)

    p = sub.add_parser("status", help="show process state")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("doctor", help="engine readiness diagnostics")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("stop", help="stop started processes")
    p.set_defaults(func=cmd_stop)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
