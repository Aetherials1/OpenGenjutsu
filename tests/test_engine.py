#!/usr/bin/env python
"""Fast structural tests - no GPU, no ComfyUI, no network.

    python tests/test_engine.py

Verifies the parts that silently break when templates are edited:
  * every workflow manifest/template parses and is self-consistent
  * every @placeholder used in a template is a real, resolvable value
  * placeholder injection preserves types
  * frames land on Wan's 4n+1 rule
  * job/api serialisation keeps the shapes the UI expects
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.engines import discover, wan_frames  # noqa: E402
from backend.engines.base import (  # noqa: E402
    JobSpec,
    TemplateEngine,
    _sam2_compatibility_problems,
    inject,
    placeholder_values,
)
from backend.media import probe, video_frame_fingerprint  # noqa: E402
from scripts.download_models import SPECS  # noqa: E402

FAILED: list[str] = []
PASSED = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global PASSED
    if ok:
        PASSED += 1
        print(f"  [pass] {label}")
    else:
        FAILED.append(label)
        print(f"  [FAIL] {label}" + (f"\n         {detail}" if detail else ""))


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_object)


def walk_strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from walk_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk_strings(value)
    elif isinstance(node, str):
        yield node


# --------------------------------------------------------------------------- #
print("\nworkflow templates")
templates = sorted(p.parent for p in (ROOT / "backend" / "workflows").glob("*/manifest.json"))
check(len(templates) >= 3, f"found {len(templates)} workflow folders")

spec = JobSpec(
    job_id="testjob",
    motion_path=Path("m.mp4"),
    character_path=Path("c.png"),
    prompt="a person dancing, cinematic",
    negative="blurry",
    width=832,
    height=480,
    fps=16.0,
    frames=65,
    steps=12,
    cfg=4.0,
    shift=5.0,
    seed=42,
    motion_ref="ogenjutsu/testjob_motion.mp4",
    character_ref="ogenjutsu/testjob_char.png",
)
values = placeholder_values(spec)

for directory in templates:
    print(f"\n  {directory.name}")
    try:
        engine = TemplateEngine(directory)
    except Exception as exc:  # noqa: BLE001
        check(False, f"{directory.name}: loads", str(exc))
        continue

    manifest = strict_json(directory / "manifest.json")
    check(all(k in manifest for k in ("id", "label", "priority", "defaults")), f"{directory.name}: manifest keys")
    check(isinstance(manifest.get("min_vram_mb"), int), f"{directory.name}: declares min_vram_mb")
    check(len(engine.required_nodes()) > 0, f"{directory.name}: lists required_nodes")
    check(manifest["id"] == directory.name, f"{directory.name}: id matches folder")
    if directory.name == "wan22-animate":
        fp8_specs = [spec for spec in SPECS["animate"] if spec.get("quant") == "fp8"]
        expected = Path(manifest["required_models"][0]).name
        downloaded = [pattern.rsplit("/", 1)[-1] for spec in fp8_specs for pattern in spec["patterns"]]
        check(expected in downloaded, "wan22-animate: downloader model matches manifest", f"{expected} not in {downloaded}")
        check(expected == "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors", "wan22-animate: matches reference Space checkpoint")

    # every placeholder resolves
    unknown: set[str] = set()
    for text in walk_strings(engine.template):
        if "@" not in text:
            continue
        for token in text.split("@")[1:]:
            name = ""
            for char in token:
                if char.isalnum() or char == "_":
                    name += char
                else:
                    break
            if name and name not in values:
                unknown.add(name)
    check(not unknown, f"{directory.name}: placeholders resolvable", f"unknown: {sorted(unknown)}")

    # injection produces native types
    graph = inject(engine.template, values)
    check(all(isinstance(node, dict) and "class_type" in node for node in graph.values()),
          f"{directory.name}: all nodes have class_type")
    if directory.name in {"wan22-animate-gguf", "wan22-animate"}:
        check(engine.replacement_mode == "sam2_mix", f"{directory.name}: marked as replacement workflow")
        output_nodes = [node_id for node_id, node in graph.items() if node.get("class_type") in {"SaveVideo", "SaveAnimatedWEBP"}]
        check(len(output_nodes) == 1, f"{directory.name}: exactly one final video output node")
    seed_ints = [
        node["inputs"]["seed"]
        for node in graph.values()
        if node.get("class_type") == "KSampler"
    ]
    check(all(isinstance(s, int) for s in seed_ints), f"{directory.name}: seed is an int")
    lengths = [
        node["inputs"]["length"]
        for node in graph.values()
        if node.get("class_type") in {"WanVaceToVideo", "WanAnimateToVideo", "WanAnimate2ToVideo"}
    ]
    check(all(isinstance(n, int) for n in lengths), f"{directory.name}: length is an int")
    check(any(node.get("class_type") in {"SaveVideo", "SaveAnimatedWEBP"} for node in graph.values()),
          f"{directory.name}: has a video output node")

    links = []
    for node_id, node in graph.items():
        for key, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                links.append((node_id, key, value))
    check(all(link[2][0] in graph for link in links), f"{directory.name}: links reference existing nodes")
    save_nodes = [node for node in graph.values() if node.get("class_type") == "SaveVideo"]
    create_nodes = {node_id for node_id, node in graph.items() if node.get("class_type") == "CreateVideo"}
    check(all((node.get("inputs") or {}).get("video", [None])[0] in create_nodes for node in save_nodes),
          f"{directory.name}: SaveVideo receives CreateVideo output")
    required = set(manifest.get("required_nodes") or [])
    present = {node.get("class_type") for node in graph.values()}
    check(required <= present, f"{directory.name}: template contains required nodes",
          f"missing: {sorted(required - present)}")

    if directory.name in {"wan22-animate-gguf", "wan22-animate"}:
        wan_node = "9"
        check(graph[wan_node]["inputs"]["pose_video"] == ["22", 0], f"{directory.name}: uses rendered pose guidance")
        check(graph[wan_node]["inputs"]["face_video"] == ["21", 1], f"{directory.name}: uses cropped face guidance")
        check(graph[wan_node]["inputs"]["background_video"] == ["27", 0], f"{directory.name}: blacked-out background guidance")
        check(graph[wan_node]["inputs"]["character_mask"] == ["26", 0], f"{directory.name}: tracked character mask")
        check(graph["24"]["inputs"]["bboxes"] == ["21", 3], f"{directory.name}: SAM2 follows detected subject boxes")
        check(graph["22"]["inputs"].get("retarget_padding") == 16, f"{directory.name}: pose retarget padding is explicit")
        check(graph["1b"]["class_type"] == "ImageScale", f"{directory.name}: reference is resized to target frame")
        check(graph["21"]["inputs"].get("retarget_image") == ["1b", 0], f"{directory.name}: detector uses resized reference")
        check(graph["5b"]["inputs"].get("image") == ["1b", 0] and graph["5b"]["inputs"].get("crop") == "none", f"{directory.name}: full reference reaches CLIP vision")
        check(graph["9"]["inputs"].get("reference_image") == ["1b", 0], f"{directory.name}: resized reference reaches Wan Animate")
        check(graph["9"]["inputs"].get("continue_motion_max_frames") == 5, f"{directory.name}: uses official continuation window")
        sam_model = graph["23"]["inputs"].get("model")
        check(sam_model == "sam2.1_hiera_base_plus.safetensors", f"{directory.name}: SAM2.1 model enables video boxes")
        check("2.1" in sam_model and graph["23"]["inputs"].get("segmentor") == "video" and "bboxes" in graph["24"]["inputs"],
              f"{directory.name}: SAM2 video+bbox combination is compatible")
        check("2.0" not in sam_model, f"{directory.name}: rejects SAM2.0 video+bbox combination")
        check(graph["6b"]["class_type"] == "LoraLoaderModelOnly", f"{directory.name}: LightX2V LoRA is wired")
        check(graph["6c"]["class_type"] == "LoraLoaderModelOnly", f"{directory.name}: relight LoRA is wired")
        check(graph["10"]["inputs"]["model"] == ["6c", 0], f"{directory.name}: sampler uses LoRA stack")
    if directory.name in {"wan22-animate-gguf", "wan22-animate"}:
        sam_spec = SPECS["animate-preprocess"][-1]
        check(sam_spec["rename"] == "sam2.1_hiera_base_plus-fp16.safetensors",
              f"{directory.name}: downloader keeps SAM2.1 fp16 filename")
        check("sam2.1_hiera_base_plus-fp16.safetensors" in sam_spec["patterns"],
              f"{directory.name}: downloader pins SAM2.1 checkpoint")
    if directory.name == "wan22-animate-gguf":
        check(graph["12"]["class_type"] == "TrimVideoLatent", "wan22-animate-gguf: trims conditioning latent")

# --------------------------------------------------------------------------- #
print("\nSAM2 compatibility guard")
unsafe_sam_graph = {
    "23": {
        "class_type": "DownloadAndLoadSAM2Model",
        "inputs": {"model": "sam2_hiera_base_plus.safetensors", "segmentor": "video"},
    },
    "24": {
        "class_type": "Sam2Segmentation",
        "inputs": {"sam2_model": ["23", 0], "bboxes": ["21", 3]},
    },
}
check(bool(_sam2_compatibility_problems(unsafe_sam_graph)), "rejects SAM2.0 video+bbox graph")
safe_sam_graph = {
    "23": {
        "class_type": "DownloadAndLoadSAM2Model",
        "inputs": {"model": "sam2.1_hiera_base_plus.safetensors", "segmentor": "video"},
    },
    "24": {
        "class_type": "Sam2Segmentation",
        "inputs": {"sam2_model": ["23", 0], "bboxes": ["21", 3]},
    },
}
check(not _sam2_compatibility_problems(safe_sam_graph), "allows SAM2.1 video+bbox graph")

# --------------------------------------------------------------------------- #
print("\nplaceholder injection")
graph = {
    "1": {"class_type": "KSampler", "inputs": {"seed": "@seed", "steps": "@steps", "cfg": "@cfg"}},
    "2": {"class_type": "LoadImage", "inputs": {"image": "@char"}},
    "3": {"class_type": "X", "inputs": {"text": "frame @seed / @width"}},
    "4": {"class_type": "Y", "inputs": {"none": "kept"}},
}
out = inject(graph, values)
check(out["1"]["inputs"]["seed"] == 42 and isinstance(out["1"]["inputs"]["seed"], int), "exact placeholder -> int")
check(out["1"]["inputs"]["cfg"] == 4.0 and isinstance(out["1"]["inputs"]["cfg"], float), "exact placeholder -> float")
check(out["2"]["inputs"]["image"] == "ogenjutsu/testjob_char.png", "exact placeholder -> str")
check(out["3"]["inputs"]["text"] == "frame 42 / 832", "embedded placeholders interpolate")
check(out["4"]["inputs"]["none"] == "kept", "plain strings untouched")
check(json.loads(json.dumps(out)) == out, "injected graph is json-serialisable")

# --------------------------------------------------------------------------- #
print("\nframes (4n+1 rule, rounded up)")
for seconds, fps, expected in [(4.0, 16.0, 65), (3.0, 16.0, 49), (1.0, 16.0, 17), (2.0, 24.0, 49)]:
    got = wan_frames(round(seconds * fps))
    check(got == expected and got % 4 == 1, f"{seconds}s @{fps}fps -> {got} frames (want {expected})")

# --------------------------------------------------------------------------- #
print("\nregistry")
engines = discover(force=True)
check("mock" in engines, "mock engine registered")
ids = {e.id for e in engines.values()}
check(ids >= {"mock", "wan21-vace"}, f"engines discovered: {sorted(ids)}")
check(all(hasattr(e, "preflight") for e in engines.values()), "all engines expose preflight")
check(engines["wan22-animate"].replacement_mode == "sam2_mix", "Mix engine is replacement-marked")
check(engines["wan-animate-2"].replacement_mode == "pose_only", "Animate 2 is explicitly pose-only")
check(engines["mock"].replacement_mode == "placeholder", "mock is explicitly placeholder-only")
mock = engines["mock"]
check(mock.defaults()["steps"] == 1, "mock defaults are cheap")
check(probe.__module__ == "backend.media", "media probe importable without a GPU")
check(video_frame_fingerprint.__module__ == "backend.media", "video fingerprint helper importable")

# --------------------------------------------------------------------------- #
print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    for name in FAILED:
        print(f"  - {name}")
    sys.exit(1)
