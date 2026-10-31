#!/usr/bin/env python
"""Generate the streamlined OpenGenjutsu Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "OpenGenjutsu_Colab.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


CELLS: list[dict] = [
    md(
        """
# OpenGenjutsu â€” one-click Colab launcher

Upload a **motion video** and a **character image** and generate a character motion-transfer video.
This notebook runs ComfyUI, OpenGenjutsu, and a Cloudflare tunnel entirely inside the Colab runtime.

## Quick start

1. Select **Runtime â†’ Change runtime type â†’ A100 GPU** (40 GB recommended).
2. Add your Hugging Face token in Colab **Secrets** as `HF_TOKEN`.
3. Run the **Settings** cell, then the **ONE-CLICK START** cell.
4. Wait for the readiness check, open the printed link, and upload your media.

The default `animate` preset uses the full **Wan 2.2 Animate 14B FP8 Mix replacement** workflow.
The A100 launcher keeps ComfyUI dynamic VRAM offloading enabled so short video latents fit safely.
It is intended for an A100-class GPU and follows the reference character-replacement pipeline:
pose/face guidance, SAM2.1 performer segmentation, background preservation, and reference-character
conditioning. Nothing is written to Google Drive unless you explicitly set `USE_DRIVE = True`.
"""
    ),
    md(
        """
## Settings

Tokens are read from Colab Secrets first, then from the values below. Add these secrets using the
key icon in the left sidebar:

- `HF_TOKEN` â€” recommended; prevents Hugging Face rate-limit errors while listing/downloading weights.
- `GH_TOKEN` â€” only needed if `REPO_URL` points to a private GitHub repository.

A Hugging Face **read** token is sufficient. Never paste a write/admin token into a shared notebook.
"""
    ),
    code(
        """
import os

# Source acquisition. ZIP-first mode avoids GitHub network access entirely.
# Upload OpenGenjutsu.zip to Colab Files before running the launcher.
REPO_URL = ""  # Deliberately disabled: no GitHub clone/fetch is performed.
ZIP_PATH = "/content/OpenGenjutsu.zip"
PROJECT_DIR = "/content/OpenGenjutsu"

# A100 defaults: full FP8 Wan 2.2 Animate Mix replacement workflow.
# Options: animate (A100/FP8), animate-q4 (lower-VRAM/GGUF), animate2, vace, all
PRESET = "animate"
USE_DRIVE = False  # Set True if you want model weights cached on Google Drive.

# Optional direct values. Prefer Colab Secrets for tokens.
HF_TOKEN = ""
GH_TOKEN = ""
OG_TOKEN = ""  # Set a fixed UI password, or leave blank for a new random password per runtime.
ENGINE = "wan22-animate"


def secret(*names: str) -> str:
    try:
        from google.colab import userdata
        for name in names:
            try:
                value = userdata.get(name)
            except Exception:
                value = ""
            if value:
                return str(value).strip()
    except Exception:
        pass
    return ""

HF_TOKEN = HF_TOKEN.strip() or secret("HF_TOKEN", "HUGGINGFACE_TOKEN")
GH_TOKEN = GH_TOKEN.strip() or secret("GH_TOKEN", "GITHUB_TOKEN")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
if GH_TOKEN:
    os.environ["GH_TOKEN"] = GH_TOKEN
os.environ["OG_PRESET"] = PRESET
os.environ["OG_ENGINE"] = ENGINE

print(f"Preset: {PRESET}")
print(f"Hugging Face token: {'configured' if HF_TOKEN else 'NOT configured (rate limits may apply)'}")
print(f"GitHub token: {'configured' if GH_TOKEN else 'not configured'}")
print(f"Google Drive: {'enabled' if USE_DRIVE else 'disabled'}")
"""
    ),
    md("## ONE-CLICK START\n\nUpload `OpenGenjutsu.zip` to the Colab Files panel first, then run this cell once after a fresh runtime. ZIP-first mode is enabled and does not contact GitHub. It installs everything, downloads the selected weights, starts the services, and prints an authenticated URL."),
    code(
        """
import base64
import os
import pathlib
import shutil
import subprocess
import sys
import time

root = pathlib.Path(PROJECT_DIR)
zip_path = pathlib.Path(ZIP_PATH)


def run(command, *, cwd=None, env=None, label=None, check=True):
    shown = " ".join(str(x) for x in command)
    print(f"\\n$ {shown}")
    result = subprocess.run(command, cwd=str(cwd) if cwd else None, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"{label or command[0]} failed with exit code {result.returncode}")
    return result.returncode


def unpack_project(src: pathlib.Path):
    print(f"Unpacking {src}...")
    tmp = pathlib.Path("/content/_ogenjutsu_unzip")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        shutil.unpack_archive(str(src), str(tmp))
    except Exception as exc:
        raise RuntimeError(f"Could not unpack {src}: {exc}. Re-upload the current OpenGenjutsu.zip.") from exc

    # Accept both layouts: backend/main.py and OpenGenjutsu/backend/main.py.
    # Some Colab uploads omit directory entries, so locate the file itself.
    main_files = [p for p in tmp.rglob("main.py") if p.parent.name.lower() == "backend"]
    if not main_files:
        sample = [str(p.relative_to(tmp)) for p in list(tmp.rglob("*"))[:20]]
        raise RuntimeError(
            "The uploaded archive does not contain backend/main.py. "
            f"Archive sample: {sample}. Re-upload the current OpenGenjutsu.zip."
        )
    source = main_files[0].parent.parent
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(source), str(root), dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Project ready: {root}")


# Obtain source. An uploaded zip is preferred when available; otherwise use git.
if zip_path.is_file():
    unpack_project(zip_path)
elif not (root / "backend" / "main.py").is_file():
    if not REPO_URL.strip():
        raise RuntimeError("Set REPO_URL or upload OpenGenjutsu.zip to the Colab Files panel.")

    root.mkdir(parents=True, exist_ok=True)
    git_env = os.environ.copy()
    git_env["GIT_TERMINAL_PROMPT"] = "0"
    if GH_TOKEN:
        # Pass the token through a temporary git config header; do not print or persist it.
        auth = base64.b64encode(f"x-access-token:{GH_TOKEN}".encode()).decode()
        git_env["GIT_CONFIG_COUNT"] = "1"
        git_env["GIT_CONFIG_KEY_0"] = "http.extraheader"
        git_env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {auth}"

    if not (root / ".git").is_dir():
        run(["git", "init"], cwd=root, env=git_env)
        run(["git", "remote", "add", "origin", REPO_URL], cwd=root, env=git_env)
    else:
        run(["git", "remote", "set-url", "origin", REPO_URL], cwd=root, env=git_env)

    fetched = run(["git", "fetch", "--depth", "1", "origin"], cwd=root, env=git_env, check=False)
    if fetched == 0:
        run(["git", "checkout", "-B", "main", "FETCH_HEAD"], cwd=root, env=git_env)
    else:
        print("GitHub source was unavailable.")

if not (root / "backend" / "main.py").is_file():
    print("\\nUpload the current OpenGenjutsu.zip in the file picker.")
    try:
        from google.colab import files
        uploaded = files.upload()
        selected = next((name for name in uploaded if name.lower().endswith((".zip", ".tar", ".tar.gz"))), None)
        if not selected:
            raise RuntimeError("Please upload OpenGenjutsu.zip")
        shutil.copy2(selected, zip_path)
        unpack_project(zip_path)
    except ImportError as exc:
        raise RuntimeError("Upload OpenGenjutsu.zip to /content, then run this cell again.") from exc

# Choose ephemeral disk by default; mount Drive only when explicitly requested.
models_dir = pathlib.Path(PROJECT_DIR) / "models"
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    models_dir = pathlib.Path("/content/drive/MyDrive/OpenGenjutsu/models")
models_dir.mkdir(parents=True, exist_ok=True)
os.environ["OG_MODELS_DIR"] = str(models_dir)

# Install system dependencies and the nodes required by Wan Animate replacement.
# The A100 uses the FP8 workflow, so the optional GGUF loader is not needed.
run([sys.executable, "scripts/colab_setup.py", "system"], cwd=root)
run([
    sys.executable, "scripts/colab_setup.py", "comfy",
    "--no-gguf", "--with-animate-preprocess",
], cwd=root)

# Download with HF_TOKEN inherited from this process. It is never included in the command line.
model_env = os.environ.copy()
if HF_TOKEN:
    model_env["HF_TOKEN"] = HF_TOKEN
    model_env["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
run([
    sys.executable, "scripts/colab_setup.py", "models",
    "--preset", PRESET, "--models-root", str(models_dir),
], cwd=root, env=model_env)

# Stop stale processes, start a clean backend, and create the tunnel.
run([sys.executable, "scripts/colab_setup.py", "stop"], cwd=root, check=False)
time.sleep(2)
import secrets
os.environ["OG_TOKEN"] = OG_TOKEN.strip() or secrets.token_urlsafe(12)
os.environ["OG_ENGINE"] = ENGINE
print("\\nStarting OpenGenjutsu. Waiting for the public link...")
serve_result = subprocess.run(
    [
        sys.executable,
        "scripts/colab_setup.py",
        "serve",
        "--profile", "big",
        "--engine", "wan22-animate",
        "--tunnel",
    ],
    cwd=str(root),
    env=os.environ.copy(),
    text=True,
    capture_output=True,
)
if serve_result.stdout:
    print(serve_result.stdout, end="")
if serve_result.stderr:
    print(serve_result.stderr, end="", file=sys.stderr)

# Print the link a second time from the tunnel log so it cannot be hidden by
# notebook output buffering or a long ComfyUI startup.
tunnel_log = pathlib.Path("/content/ogenjutsu-logs/tunnel.log")
if tunnel_log.exists():
    import re
    tunnel_text = tunnel_log.read_text(encoding="utf-8", errors="replace")
    tunnel_urls = re.findall(r"https://[A-Za-z0-9.-]+\\.trycloudflare\\.com", tunnel_text)
    if tunnel_urls:
        public_url = tunnel_urls[-1]
        print("\\n" + "=" * 72)
        print("OPEN THIS LINK:")
        print(f"{public_url}/?token={os.environ.get('OG_TOKEN', '')}")
        print("=" * 72)
    else:
        print("\\nNo Cloudflare URL was found. Last tunnel log output:")
        print(tunnel_text[-4000:])

if serve_result.returncode != 0:
    raise RuntimeError(f"serve failed with exit code {serve_result.returncode}")

print("\\nRunning readiness check...")
run([sys.executable, "scripts/colab_setup.py", "doctor"], cwd=root, env=os.environ.copy(), check=False)
"""
    ),
    md("## Reconnect / health check\n\nUse this after a temporary tunnel disconnect. It does not redownload weights."),
    code(
        """
import os, pathlib, subprocess, sys
root = pathlib.Path(PROJECT_DIR)
subprocess.run([sys.executable, "scripts/colab_setup.py", "tunnel"], cwd=root, env=os.environ.copy())
subprocess.run([sys.executable, "scripts/colab_setup.py", "doctor"], cwd=root, env=os.environ.copy())
"""
    ),
    md("## Stop and free the GPU"),
    code(
        """
import pathlib, subprocess, sys
root = pathlib.Path(PROJECT_DIR)
subprocess.run([sys.executable, "scripts/colab_setup.py", "stop"], cwd=root)
"""
    ),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100", "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


def main() -> None:
    OUT.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
