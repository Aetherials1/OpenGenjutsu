#!/usr/bin/env python
"""Download the model files ComfyUI needs for each OpenGenjutsu engine.

Files are pulled from Hugging Face into a cache directory (on Colab: ephemeral
storage by default, or Google Drive only when explicitly selected) and then
symlinked into ``ComfyUI/models/<type>/``. Set ``HF_TOKEN`` or
``HUGGINGFACE_HUB_TOKEN`` to avoid anonymous API rate limits.

Exact filenames change upstream, so every spec is a glob matched against the real
repository file list. Use ``--dry-run`` to see what would be fetched.

    python scripts/download_models.py --preset vace --dry-run
    python scripts/download_models.py --preset animate --quant q4
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# repo -> destination folder inside ComfyUI/models
SPECS: dict[str, list[dict]] = {
    "shared": [
        {
            "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "patterns": ["split_files/vae/wan_2.1_vae.safetensors"],
            "dest": "vae",
            "label": "Wan 2.1 VAE",
        },
        {
            "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "patterns": ["split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
            "dest": "text_encoders",
            "label": "umt5-xxl text encoder (fp8)",
        },
        {
            "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "patterns": ["split_files/clip_vision/clip_vision_h.safetensors"],
            "dest": "clip_vision",
            "label": "CLIP vision H",
        },
    ],
    "vace": [
        {
            "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "patterns": [
                "split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors",
                "split_files/diffusion_models/*vace*1.3B*fp16*.safetensors",
            ],
            "dest": "diffusion_models",
            "label": "Wan 2.1 VACE 1.3B",
        },
    ],
    "animate": [
        {
            "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "patterns": [
                "split_files/diffusion_models/*animate*14B*fp8*.safetensors",
                "split_files/diffusion_models/*animate*14B*.safetensors",
            ],
            "dest": "diffusion_models",
            "label": "Wan 2.2 Animate 14B (fp8)",
            "quant": "fp8",
        },
        {
            "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "patterns": [
                "split_files/loras/*animate*relight*.safetensors",
                "split_files/loras/*relight*.safetensors",
            ],
            "dest": "loras",
            "label": "Animate relight LoRA",
            "optional": True,
        },
        {
            "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "patterns": ["split_files/loras/*animate*lightning*.safetensors"],
            "dest": "loras",
            "label": "Animate lightning LoRA (4-8 steps)",
            "optional": True,
        },
    ],
    "animate-q4": [
        {
            "repo": "QuantStack/Wan2.2-Animate-14B-GGUF",
            "patterns": ["*Q4_K_M*.gguf", "*Q4*.gguf"],
            "dest": "diffusion_models",
            "label": "Wan 2.2 Animate 14B Q4_K_M (GGUF)",
            "quant": "q4",
            "required": True,
        },
    ],
    "animate2": [
        {
            "repo": "Comfy-Org/Wan-Animate-2",
            "patterns": [
                "diffusion_models/wan_animate_2_int8_convrot.safetensors",
                "diffusion_models/wan_animate_2_distill_int8_convrot.safetensors",
                "diffusion_models/*animate_2*.safetensors",
            ],
            "dest": "diffusion_models",
            "label": "Wan Animate 2 (14B DiT SOTA)",
        },
    ],
}

PRESETS = {
    "vace": ["shared", "vace"],
    "animate": ["shared", "animate"],
    "animate-q4": ["shared", "animate-q4"],
    "animate2": ["shared", "animate2"],
    "all": ["shared", "vace", "animate2", "animate-q4", "animate"],
}


def log(msg: str) -> None:
    print(f"[models] {msg}", flush=True)


def repo_files(repo_id: str) -> list[str]:
    """List every file in a HF repo (works for public repos, no auth needed)."""
    from huggingface_hub import HfApi

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token or None)
    try:
        return [item.rfilename for item in api.list_repo_tree(repo_id, recursive=True) if hasattr(item, "rfilename")]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"cannot list {repo_id}: {exc}. Add a Hugging Face read token as "
            "HF_TOKEN in Colab Secrets and rerun this cell."
        ) from exc


def match(repo_id: str, patterns: list[str], cache: dict[str, list[str]]) -> list[str]:
    files = cache.get(repo_id)
    if files is None:
        files = repo_files(repo_id)
        cache[repo_id] = files
    hits: list[str] = []
    for pattern in patterns:
        for name in files:
            if fnmatch.fnmatch(name, pattern) and name not in hits:
                hits.append(name)
    return hits


def link_into(source: Path, models_root: Path, dest: str) -> Path:
    target_dir = models_root / dest
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == source.resolve():
                return target
        except OSError:
            pass
        if target.is_symlink() or target.is_file():
            target.unlink()
    try:
        target.symlink_to(source)
        log(f"linked {dest}/{source.name}")
    except OSError:
        import shutil

        log(f"symlink unavailable, copying {source.name} -> {dest}/")
        shutil.copy2(source, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Wan models for OpenGenjutsu")
    parser.add_argument("--preset", default=os.getenv("OG_PRESET", "animate2"),
                        help=f"weights preset (default: animate2, choices: {sorted(PRESETS)})")
    parser.add_argument("--models-root", default=os.getenv("OG_MODELS_DIR", ""),
                        help="cache directory (default: <project>/models)")
    parser.add_argument("--comfy-root", default=os.getenv("OG_COMFY_DIR", ""),
                        help="ComfyUI checkout (default: <project>/../ComfyUI)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quant", default="", choices=["", "fp8", "q4"],
                        help="skip specs whose quant does not match")
    args = parser.parse_args()

    preset = (args.preset or "").strip()
    if not preset or preset in ("{PRESET}", "$PRESET", "${PRESET}", "$OG_PRESET", "${OG_PRESET}"):
        preset = os.getenv("OG_PRESET", "animate2")
    if preset not in PRESETS:
        log(f"Unrecognized preset '{preset}', defaulting to 'animate2'. Available presets: {sorted(PRESETS)}")
        preset = "animate2"

    cache_root = Path(args.models_root or PROJECT_ROOT / "models").expanduser()
    comfy_root = Path(args.comfy_root or PROJECT_ROOT.parent / "ComfyUI").expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    specs: list[dict] = []
    for key in PRESETS[preset]:
        specs.extend(SPECS[key])

    files_cache: dict[str, list[str]] = {}
    failures: list[str] = []
    planned = 0

    for spec in specs:
        if args.quant and spec.get("quant") and spec["quant"] != args.quant:
            continue
        repo = spec["repo"]
        label = spec.get("label", repo)
        hits = match(repo, spec["patterns"], files_cache)
        if not hits:
            message = f"{label}: no file in {repo} matched {spec['patterns']}"
            if spec.get("optional"):
                log(f"skip (optional) {message}")
                continue
            failures.append(message)
            log(f"ERROR {message}")
            continue

        name = hits[0]
        planned += 1
        size_note = ""
        log(f"{label}\n    {repo}/{name}{size_note}")

        if args.dry_run:
            continue

        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=repo,
                filename=name,
                local_dir=str(cache_root / repo.replace("/", "__")),
                token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None,
                resume_download=True,
            )
            link_into(Path(path), comfy_root / "models", spec["dest"])
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: {exc}")
            log(f"ERROR {exc}")

    if args.dry_run:
        log(f"dry run: {planned} file(s) would be fetched into {cache_root}")
        return 0 if not failures else 1

    if failures:
        log(f"{len(failures)} problem(s):")
        for item in failures:
            log(f"  - {item}")
        log("Upstream may have renamed files. Edit SPECS in scripts/download_models.py.")
        return 1

    log("done - models are linked into ComfyUI/models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
