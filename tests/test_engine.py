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
    inject,
    placeholder_values,
)
from backend.media import probe  # noqa: E402

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

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    check(all(k in manifest for k in ("id", "label", "priority", "defaults")), f"{directory.name}: manifest keys")
    check(isinstance(manifest.get("min_vram_mb"), int), f"{directory.name}: declares min_vram_mb")
    check(len(engine.required_nodes()) > 0, f"{directory.name}: lists required_nodes")
    check(manifest["id"] == directory.name, f"{directory.name}: id matches folder")

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
    seed_ints = [
        node["inputs"]["seed"]
        for node in graph.values()
        if node.get("class_type") == "KSampler"
    ]
    check(all(isinstance(s, int) for s in seed_ints), f"{directory.name}: seed is an int")
    lengths = [
        node["inputs"]["length"]
        for node in graph.values()
        if node.get("class_type") in {"WanVaceToVideo", "WanAnimateToVideo"}
    ]
    check(all(isinstance(n, int) for n in lengths), f"{directory.name}: length is an int")
    check(any(node.get("class_type") in {"SaveVideo", "SaveAnimatedWEBP"} for node in graph.values()),
          f"{directory.name}: has a video output node")

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
mock = engines["mock"]
check(mock.defaults()["steps"] == 1, "mock defaults are cheap")
check(probe.__module__ == "backend.media", "media probe importable without a GPU")

# --------------------------------------------------------------------------- #
print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    for name in FAILED:
        print(f"  - {name}")
    sys.exit(1)
