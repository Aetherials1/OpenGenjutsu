#!/usr/bin/env python
"""Generate ``notebooks/OpenGenjutsu_Colab.ipynb``.

Keeping the notebook generated means the cells stay reviewable as plain Python
instead of a 900-line JSON blob. Re-run after editing cells:

    python notebooks/build_notebook.py
"""

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


CELLS: list[dict] = []

CELLS.append(
    md(
        """
# OpenGenjutsu — motion transfer on this Colab GPU

Upload a **motion video** and a **character image**, get back your character
performing that motion, with an optional carry-over of the clip's original audio.

This notebook starts three things inside the runtime:

| | |
|---|---|
| **ComfyUI** `:8188` | the diffusion engine (localhost only) |
| **OpenGenjutsu backend** `:7860` | API, job queue, all ffmpeg / audio work |
| **cloudflared** | a public HTTPS URL for the backend |

Then you open that URL in any browser — this machine, your phone, whatever.

**Before you start:** `Runtime → Change runtime type → T4 GPU`.
Run the cells in order. Re-running any cell is safe.
"""
    )
)

CELLS.append(
    md(
        """
## 1 · Settings

* `REPO_URL` — git URL for this project (or leave empty if using uploaded zip).
* `GH_TOKEN` — optional GitHub Personal Access Token (or add `GH_TOKEN` in Colab Secrets 🔑).
* `ZIP_PATH` — path for uploaded `OpenGenjutsu.zip` (drag & drop into Colab files panel).
* `PRESET` — which weights to fetch:
  * `animate2` — **Wan-Animate-2 14B DiT (NEW SOTA)**: Alibaba's newest model, distilled 10-step, direct video-in-transformer motion transfer with stable hands and faces *(recommended)*
  * `vace` — Wan 2.1 VACE 1.3B, ≈11 GB, works everywhere, 2–4 min per 4 s clip *(lightest & fastest)*
  * `animate-q4` — Wan 2.2 Animate 14B GGUF Q4, ≈14 GB, low-VRAM 14B quant
  * `animate` — Wan 2.2 / Animate 14B fp8, needs >22 GB VRAM (for A100/L4 on Colab Pro)
* `USE_DRIVE` — cache weights in Google Drive (leave False if your Drive is full; weights download straight to Colab's ephemeral disk instead).
* `OG_TOKEN` — leave empty and a random one is generated. **Keep it.** Anyone with
  the tunnel URL + token can spend your GPU quota.
"""
    )
)

CELLS.append(
    code(
        """
import os

REPO_URL   = "https://github.com/Aetherials1/OpenGenjutsu.git"  # or leave empty if using zip
GH_TOKEN   = ""                                    # GitHub token if repo is private (or Colab Secrets)
ZIP_PATH   = "/content/OpenGenjutsu.zip"           # drag & drop OpenGenjutsu.zip into Colab files
PROJECT_DIR = "/content/OpenGenjutsu"

PRESET     = "animate2"      # "animate2" (New SOTA DiT) | "vace" (fastest, light) | "animate-q4" | "animate"
USE_DRIVE  = False           # False = use Colab's free ~100 GB disk (leaves your Google Drive untouched!)
OG_TOKEN   = ""              # empty -> random token, printed below
ENGINE     = "auto"          # "auto" | "wan-animate-2" | "wan21-vace" | "wan22-animate-gguf" | "wan22-animate" | "mock"

os.environ["OG_PRESET"] = PRESET
"""
    )
)

CELLS.append(
    md("## 2 · Check the GPU")
)

CELLS.append(
    code(
        """
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    total_gb = total / 1024**3
    print(f"torch {torch.__version__} | {name} | {total_gb:.1f} GB total, {free / 1024**3:.1f} GB free")
    if total_gb >= 20:
        print(f"🚀 Colab Pro / High-VRAM GPU detected ({name})! Full-speed inference + 720p HD unlocked.")
    else:
        print(f"⚡ Standard GPU detected ({name}). Memory-efficient mode enabled.")
else:
    print("NO CUDA GPU. Runtime -> Change runtime type -> T4 GPU, then re-run.")
"""
    )
)

CELLS.append(
    md(
        """
## 3 · Get the project

Either clones `REPO_URL`, unpacks `ZIP_PATH`, or reuses an existing
`/content/OpenGenjutsu`.
"""
    )
)

CELLS.append(
    code(
        """
import os, pathlib, shutil, subprocess

root = pathlib.Path(PROJECT_DIR)
zip_path = pathlib.Path(ZIP_PATH)

def unpack_zip(src: pathlib.Path, dst: pathlib.Path):
    print(f"Unpacking {src} -> {dst}...")
    tmp = pathlib.Path("/content/_og_unzip")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.unpack_archive(str(src), str(tmp))
    matches = [p for p in tmp.rglob("backend") if (p / "main.py").is_file()]
    if not matches:
        raise RuntimeError(f"Could not find backend/main.py inside {src}")
    source_dir = matches[0].parent
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(source_dir), str(dst), dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("Project successfully unpacked from zip!")

# Check if already installed
if (root / "backend" / "main.py").is_file():
    print(f"OpenGenjutsu already present at {PROJECT_DIR}.")
    if (root / ".git").is_dir():
        print("Pulling latest git changes...")
        subprocess.run(["git", "-C", PROJECT_DIR, "pull", "--ff-only"], capture_output=True)
else:
    token = (GH_TOKEN or "").strip()
    if not token:
        try:
            from google.colab import userdata
            token = (userdata.get("GH_TOKEN") or userdata.get("GITHUB_TOKEN") or "").strip()
        except Exception:
            token = ""

    obtained = False

    # Priority 1: If an uploaded zip exists and no GitHub token is given, unpack immediately
    if zip_path.is_file() and not token:
        print(f"Found uploaded zip at {zip_path}!")
        unpack_zip(zip_path, root)
        obtained = (root / "backend" / "main.py").is_file()

    # Priority 2: Git fetch if REPO_URL is provided and not yet obtained
    if not obtained and REPO_URL.strip():
        clone_url = REPO_URL.strip()
        if token and "github.com" in clone_url and "@" not in clone_url:
            clone_url = clone_url.replace("https://github.com/", f"https://{token}@github.com/")
        if not clone_url.endswith(".git") and "github.com" in clone_url:
            clone_url += ".git"

        print(f"Fetching repository from {REPO_URL}...")
        root.mkdir(parents=True, exist_ok=True)
        if not (root / ".git").is_dir():
            subprocess.run(["git", "init"], cwd=PROJECT_DIR, capture_output=True)
            subprocess.run(["git", "remote", "add", "origin", clone_url], cwd=PROJECT_DIR, capture_output=True)
        else:
            subprocess.run(["git", "remote", "set-url", "origin", clone_url], cwd=PROJECT_DIR, capture_output=True)

        fetch_res = subprocess.run(["git", "fetch", "--depth", "1", "origin"], cwd=PROJECT_DIR, capture_output=True, text=True)
        if fetch_res.returncode == 0:
            checkout_res = subprocess.run(["git", "checkout", "-B", "main", "FETCH_HEAD"], cwd=PROJECT_DIR, capture_output=True, text=True)
            if checkout_res.returncode == 0 and (root / "backend" / "main.py").is_file():
                print("Cloned and checked out successfully.")
                obtained = True
        else:
            print("Git fetch failed (repository may be private or inaccessible).")

    # Priority 3: Fall back to zip if git failed and zip exists
    if not obtained and zip_path.is_file():
        print(f"Falling back to uploaded zip at {zip_path}...")
        unpack_zip(zip_path, root)
        obtained = (root / "backend" / "main.py").is_file()

    # If still not obtained, display clear instructions
    if not (root / "backend" / "main.py").is_file():
        print("\\n" + "=" * 68)
        print("❌ Could not obtain OpenGenjutsu source code.")
        print("\\nFix this with ONE of these simple options:")
        print("  Option 1 (Easiest - No token needed):")
        print("    Drag & drop 'OpenGenjutsu.zip' from your computer into the Colab")
        print(f"    file browser (left panel) at {ZIP_PATH}, then run this cell again.")
        print("  Option 2 (Using GitHub Personal Access Token):")
        print("    Set GH_TOKEN = 'ghp_yourToken' in Cell 1, or add a secret")
        print("    named 'GH_TOKEN' in Colab Secrets (the key icon 🔑 on left).")
        print("  Option 3 (Make Repository Public):")
        print("    In GitHub repository Settings -> Change visibility to Public.")
        print("=" * 68 + "\\n")
        raise SystemExit("Missing OpenGenjutsu files.")

%cd {PROJECT_DIR}
print("project ready:", os.getcwd())
"""
    )
)

CELLS.append(
    md("## 4 · System packages (ffmpeg)")
)

CELLS.append(
    code(
        """
!python scripts/colab_setup.py system
"""
    )
)

CELLS.append(
    md(
        """
## 5 · Model cache on Drive

Weights are big and Drive persists between sessions. Downloads go to
`My Drive/OpenGenjutsu/models` and are symlinked into ComfyUI, so nothing is
duplicated.
"""
    )
)

CELLS.append(
    code(
        """
import os

MODELS_DIR = "/content/OpenGenjutsu/models"
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    MODELS_DIR = "/content/drive/MyDrive/OpenGenjutsu/models"

os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["OG_MODELS_DIR"] = MODELS_DIR
print("model cache:", MODELS_DIR)
"""
    )
)

CELLS.append(
    md(
        """
## 6 · Install ComfyUI

Clones/updates ComfyUI and installs its requirements. Adds the GGUF loader when
`PRESET` is `animate-q4`.
"""
    )
)

CELLS.append(
    code(
        """
import os

cmd = "python scripts/colab_setup.py comfy --with-gguf"
print("$", cmd)
!{cmd}
"""
    )
)

CELLS.append(
    md(
        """
## 7 · Download weights

This is the long cell on a fresh runtime (≈11–14 GB). With `USE_DRIVE` the next
session skips it. `--dry-run` is available if you want to see what would be
fetched.
"""
    )
)

CELLS.append(
    code(
        """
import os

preset = globals().get("PRESET") or os.environ.get("OG_PRESET", "animate2")
models_dir = globals().get("MODELS_DIR") or os.environ.get("OG_MODELS_DIR", "/content/OpenGenjutsu/models")

cmd = f'python scripts/colab_setup.py models --preset {preset} --models-root "{models_dir}"'
print("$", cmd)
!{cmd}
"""
    )
)

CELLS.append(
    md(
        """
## 8 · Launch OpenGenjutsu (ComfyUI + Backend + Tunnel)

Boots ComfyUI, starts the OpenGenjutsu backend, and opens a secure Cloudflare tunnel.
Your public web app link and Access Token will be printed directly in this cell when ready!
"""
    )
)

CELLS.append(
    code(
        """
import os, secrets

os.environ["OG_ENGINE"] = ENGINE
if not os.environ.get("OG_TOKEN"):
    os.environ["OG_TOKEN"] = OG_TOKEN or secrets.token_urlsafe(12)

print("ACCESS TOKEN:", os.environ["OG_TOKEN"])
!python scripts/colab_setup.py serve --tunnel
"""
    )
)

CELLS.append(
    md(
        """
## 9 · Tunnel / Reconnect (Optional)

If you ever need to view your active link again or restart the tunnel, run this cell.
"""
    )
)

CELLS.append(
    code(
        """
!python scripts/colab_setup.py tunnel
"""
    )
)

CELLS.append(
    md(
        """
## 10 · Health & Doctor Check

Verifies that all AI engines (Wan-Animate-2, VACE, etc.) are healthy and ready to render.
"""
    )
)

CELLS.append(
    code(
        """
!python scripts/colab_setup.py doctor
"""
    )
)

CELLS.append(
    code(
        """
!python scripts/smoke_test.py --url http://127.0.0.1:7860 --token "$OG_TOKEN"
"""
    )
)

CELLS.append(
    md(
        """
## Tips

* **Keep the Colab tab open.** Closing it recycles the runtime and kills the
  tunnel. A new session = re-run cells 3–9 (models are cached in Drive).
* **Length is the main cost.** 3–4 s at 832×480 is the sweet spot on a T4.
* **Steps** — 16 is a good VACE default; 8–12 is noticeably faster with little
  loss. CFG 1.0 with a distilled LoRA is fastest where supported.
* **Prompt** describes style/lighting; identity comes from the image, motion from
  the video. Keep it short.
* **Audio** — the toggle attaches the motion clip's own track, trimmed or padded
  to the render's exact length. If the clip has no audio track you'll get a
  warning before rendering.
* **Compare tab** plays source and result side by side.

## Troubleshooting

* `doctor` blocked on a missing node → `docs/TROUBLESHOOTING.md` (or pick another
  engine in the UI).
* CUDA OOM → shorter clip, 832×480, or `PRESET = "vace"`.
* No tunnel URL → re-run cell 9; if DNS is filtered, try a phone hotspot.
* Everything is logged in `/content/ogenjutsu-logs/`.
"""
    )
)

CELLS.append(
    md("## Stop everything / free the GPU")
)

CELLS.append(
    code(
        """
!python scripts/colab_setup.py stop
!python scripts/colab_setup.py status
"""
    )
)


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "toc_visible": True,
        },
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
