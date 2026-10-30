"""Engine layer: each engine turns a job request into a rendered video file.

Two flavours exist:

* :class:`TemplateEngine` drives ComfyUI from a JSON workflow template found in
  ``backend/workflows/<engine>/``. Templates contain placeholders such as
  ``"@motion"`` or ``"@seed"`` which are substituted at run time, so the user can
  edit/replace a workflow in the ComfyUI UI and drop it in without touching code.
* :class:`~backend.engines.mock.MockEngine` renders a stand-in clip with ffmpeg
  so the whole app (UI, queue, audio muxing, downloads) works without a GPU.
"""

from __future__ import annotations

import json
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import settings
from ..media import MediaError, ensure_mp4, probe

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


class EngineError(RuntimeError):
    """A generation could not be completed."""


def wan_frames(frames: int | float) -> int:
    """Snap a frame count to Wan's requirement of 4n+1 (rounded up, min 5).

    Rounding up keeps the render at least as long as the user asked for, which is
    far less surprising than silently returning a shorter clip.
    """
    value = max(5, int(frames))
    return -(-(value - 1) // 4) * 4 + 1


@dataclass
class JobSpec:
    """Everything an engine needs to render one clip (engine agnostic)."""

    job_id: str
    motion_path: Path
    character_path: Path
    prompt: str = ""
    negative: str = ""
    width: int = 832
    height: int = 480
    fps: float = 16.0
    # Wan Animate is generated at its native 16 fps.  ``fps`` is the requested
    # delivery rate; the render is retimed after sampling when it is higher.
    model_fps: float = 16.0
    frames: int = 49
    steps: int = 6
    cfg: float = 1.0
    shift: float = 5.0
    seed: int = 0
    loop: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    # filled in by the engine once inputs are staged for ComfyUI
    motion_ref: str = ""
    character_ref: str = ""

    # probed source metadata
    motion_info: dict[str, Any] = field(default_factory=dict)
    character_info: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.model_fps > 0:
            return self.frames / self.model_fps
        return float(self.motion_info.get("duration") or 0.0)


@dataclass
class EngineResult:
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)


class Engine(ABC):
    id: str = "base"
    label: str = "Base"
    description: str = ""
    requires_comfy: bool = False
    requires_gpu: bool = False
    maturity: str = "stable"  # stable | experimental | test
    notes: str = ""

    def defaults(self) -> dict[str, Any]:
        return {
            "width": settings.default_width,
            "height": settings.default_height,
            "fps": settings.default_fps,
            "frames": wan_frames(
                settings.default_fps * min(4.0, settings.default_max_seconds)
            ),
            "steps": 6,
            "cfg": 1.0,
            "max_seconds": settings.default_max_seconds,
        }

    def preflight(self) -> dict[str, Any]:
        """Return {"ready": bool, "problems": [...], "warnings": [...]}."""
        return {"ready": True, "problems": [], "warnings": []}

    def info(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = state or self.preflight()
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "maturity": self.maturity,
            "notes": self.notes,
            "requires_comfy": self.requires_comfy,
            "requires_gpu": self.requires_gpu,
            "defaults": self.defaults(),
            "ready": state["ready"],
            "problems": state["problems"],
            "warnings": state["warnings"],
        }

    @abstractmethod
    def run(
        self,
        spec: JobSpec,
        workdir: Path,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> EngineResult:
        ...


# --------------------------------------------------------------------------- #
# ComfyUI workflow-template engine
# --------------------------------------------------------------------------- #

_PLACEHOLDER = re.compile(r"@([a-z_]+)")


def placeholder_values(spec: JobSpec) -> dict[str, Any]:
    """Values available to workflow templates."""
    return {
        "char": spec.character_ref,
        "motion": spec.motion_ref,
        "prompt": spec.prompt,
        "negative": spec.negative,
        "seed": int(spec.seed),
        "steps": int(spec.steps),
        "cfg": float(spec.cfg),
        "shift": float(spec.shift),
        "width": int(spec.width),
        "height": int(spec.height),
        "fps": float(spec.model_fps),
        "output_fps": float(spec.fps),
        "frames": int(spec.frames),
        "length": int(spec.frames),
        "duration": float(spec.duration),
        "prefix": f"og_{spec.job_id}",
        "job": spec.job_id,
    }


def inject(node: Any, values: dict[str, Any]) -> Any:
    """Recursively substitute ``"@name"`` placeholders in a workflow graph.

    A value that is exactly one placeholder keeps its native type (int stays int).
    Placeholders embedded in a longer string are string-formatted.
    """
    if isinstance(node, dict):
        return {k: inject(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [inject(v, values) for v in node]
    if isinstance(node, str) and "@" in node:
        exact = _PLACEHOLDER.fullmatch(node)
        if exact:
            key = exact.group(1)
            if key in values:
                return values[key]
            return node
        return _PLACEHOLDER.sub(
            lambda m: str(values.get(m.group(1), m.group(0))), node
        )
    return node


def _mesh_models(graph: dict[str, Any]) -> set[str]:
    """Collect every model filename referenced by loader nodes."""
    names: set[str] = set()
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        ctype = str(node.get("class_type") or "")
        if "Loader" in ctype or "Load" in ctype:
            for key in ("ckpt_name", "unet_name", "vae_name", "clip_name", "lora_name"):
                value = inputs.get(key)
                if isinstance(value, str) and not value.startswith("@"):
                    names.add(value)
    return names


class TemplateEngine(Engine):
    requires_comfy = True
    requires_gpu = True

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.manifest: dict[str, Any] = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        self.template: dict[str, Any] = json.loads(
            (directory / "template.json").read_text(encoding="utf-8")
        )
        self.id = self.manifest.get("id", directory.name)
        self.label = self.manifest.get("label", self.id)
        self.description = self.manifest.get("description", "")
        self.maturity = self.manifest.get("maturity", "experimental")
        self.notes = self.manifest.get("notes", "")

    # ---------------------------------------------------------------- config
    def defaults(self) -> dict[str, Any]:
        base = super().defaults()
        manifest_defaults = self.manifest.get("defaults") or {}
        base.update(
            {k: v for k, v in manifest_defaults.items() if k != "frames"}
        )
        if "frames" in manifest_defaults:
            base["frames"] = wan_frames(manifest_defaults["frames"])
        if "model_fps" in manifest_defaults:
            base["model_fps"] = float(manifest_defaults["model_fps"])
        if "output_fps" in manifest_defaults:
            base["fps"] = float(manifest_defaults["output_fps"])
        base["max_seconds"] = float(
            self.manifest.get("max_seconds", settings.default_max_seconds)
        )
        return base

    def required_nodes(self) -> list[str]:
        return list(self.manifest.get("required_nodes") or [])

    def required_models(self) -> list[str]:
        return list(self.manifest.get("required_models") or [])

    def _client(self):
        from ..comfy import ComfyClient

        return ComfyClient()

    def preflight(self) -> dict[str, Any]:
        problems: list[str] = []
        warnings: list[str] = []

        client = self._client()
        if not client.ping():
            return {
                "ready": False,
                "problems": [
                    f"ComfyUI is not reachable at {settings.comfy_url}. "
                    "Start it with the Colab notebook or `scripts/run_comfy.ps1`."
                ],
                "warnings": warnings,
            }

        # Hardware gate: a 14B fp8 model on a 16 GB card will OOM rather than
        # render, so refuse early with an actionable message.
        min_vram = int(self.manifest.get("min_vram_mb") or 0)
        if min_vram:
            from ..comfy import gpu_info

            gpu = gpu_info()
            if gpu.get("available") and gpu.get("vram_total_mb", 0) < min_vram:
                detail = (
                    f"needs ~{min_vram // 1024} GB VRAM but this GPU has "
                    f"{gpu['vram_total_mb'] // 1024} GB"
                )
                if settings.ignore_vram:
                    warnings.append(
                        f"{detail} - OG_IGNORE_VRAM is set, so it will be attempted "
                        "anyway (expect OOM or heavy offloading)."
                    )
                else:
                    problems.append(
                        f"{detail} - pick a lighter engine, or set OG_IGNORE_VRAM=1 to try "
                        "anyway (see docs/WORKFLOWS.md)."
                    )

        try:
            info = client.object_info()
        except Exception as exc:
            return {
                "ready": False,
                "problems": [f"ComfyUI did not answer /object_info: {exc}"],
                "warnings": warnings,
            }

        missing = [n for n in self.required_nodes() if n not in info]
        if missing:
            problems.append(
                "ComfyUI is missing node(s): "
                + ", ".join(missing)
                + ". Update ComfyUI or see docs/TROUBLESHOOTING.md."
            )

        # Verify every input our template sets actually exists on the installed
        # node - this catches ComfyUI version drift before a render is queued.
        # Newer ComfyUI exposes the native schema as a nested dict; older builds
        # return the legacy ``required``/``optional`` shape directly.
        ignored_inputs = {"frame_load_cap", "skip_first_frames", "select_every_nth"}
        for node_id, node in self.template.items():
            if not isinstance(node, dict):
                continue
            ctype = node.get("class_type")
            spec = info.get(str(ctype))
            if not spec:
                continue  # already reported above when listed in required_nodes
            declared = spec.get("input") or {}
            valid = set(declared.get("required") or {}) | set(declared.get("optional") or {})
            if not valid:
                continue
            for key in (node.get("inputs") or {}):
                if key in ignored_inputs:
                    continue
                if key not in valid:
                    problems.append(
                        f"{ctype} (#{node_id}) has no input '{key}' in this ComfyUI "
                        f"build - template needs updating ({len(problems) + 1})"
                    )
                else:
                    val = (node.get("inputs") or {}).get(key)
                    if isinstance(val, str) and not val.startswith("@") and key in ("unet_name", "ckpt_name"):
                        input_def = (declared.get("required") or {}).get(key) or (declared.get("optional") or {}).get(key)
                        if isinstance(input_def, (list, tuple)) and input_def and isinstance(input_def[0], list):
                            choices = input_def[0]
                            if choices and val not in choices:
                                problems.append(
                                    f"Model '{val}' is not installed in ComfyUI models folder."
                                )
                            elif not choices:
                                problems.append(
                                    f"No model weights installed for {ctype} (folder is empty in ComfyUI)."
                                )

        # The ComfyUI model dropdown can be empty when the model is missing.
        # The explicit filename check above reports that as a useful preflight error.
        models = _mesh_models(self.template)
        if models:
            warnings.append(
                "Workflow expects these model files in ComfyUI: "
                + ", ".join(sorted(models))
            )

        # Keep the payload readable when something is badly out of date.
        if len(problems) > 8:
            extra = len(problems) - 8
            problems = problems[:8] + [f"... and {extra} more template/ComfyUI mismatches"]

        return {"ready": not problems, "problems": problems, "warnings": warnings}

    # ------------------------------------------------------------------- run
    def run(
        self,
        spec: JobSpec,
        workdir: Path,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> EngineResult:
        from ..comfy import ComfyClient, ComfyError, new_client_id

        client = ComfyClient()
        if not client.ping():
            raise EngineError(
                f"ComfyUI is not reachable at {settings.comfy_url}."
            )

        def report(value: float, stage: str) -> None:
            if progress:
                progress(max(0.0, min(0.99, value)), stage)

        report(0.01, "staging inputs")
        try:
            spec.character_ref = client.stage_input(
                spec.character_path, f"{spec.job_id}_char{spec.character_path.suffix}"
            )
            spec.motion_ref = client.stage_input(
                spec.motion_path, f"{spec.job_id}_motion{spec.motion_path.suffix}"
            )
        except ComfyError as exc:
            client.clear_staging()
            raise EngineError(str(exc)) from exc

        try:
            report(0.03, "building workflow")
            graph = inject(self.template, placeholder_values(spec))
            for n in graph.values():
                if isinstance(n, dict) and n.get("class_type") == "LoadVideo":
                    inputs = n.get("inputs", {})
                    for dep in ("frame_load_cap", "skip_first_frames", "select_every_nth"):
                        inputs.pop(dep, None)

            client_id = new_client_id()
            try:
                prompt_id = client.submit(graph, client_id)
            except ComfyError as exc:
                raise EngineError(str(exc)) from exc

            report(0.05, "queued")
            try:
                entry = client.wait(
                    prompt_id,
                    client_id,
                    on_progress=lambda v, s: report(v, _stage_label(s)),
                    should_cancel=cancel,
                    timeout=settings.job_timeout_s,
                )
            except ComfyError as exc:
                if "__canceled__" in str(exc):
                    raise EngineError("canceled") from exc
                raise EngineError(str(exc)) from exc

            report(0.97, "retrieving render")
            try:
                raw = client.fetch_output(entry, workdir / "raw")
            except ComfyError as exc:
                raise EngineError(str(exc)) from exc
        finally:
            # Inputs are consumed once the prompt has finished or failed.
            client.clear_staging()

        render = workdir / "render.mp4"
        try:
            if raw.suffix.lower() == ".mp4":
                shutil.move(str(raw), render)
            else:
                ensure_mp4(raw, render, fps=spec.model_fps)
        except MediaError as exc:
            raise EngineError(f"could not convert render to mp4: {exc}") from exc

        info = probe(render)
        report(0.99, "finalising")
        return EngineResult(
            path=render,
            meta={
                "prompt_id": prompt_id,
                "width": info["width"],
                "height": info["height"],
                "fps": round(info["fps"], 3),
                "duration": round(float(info["duration"]), 3),
                "frames": info["nb_frames"],
                "engine": self.id,
            },
        )


def _stage_label(raw: str) -> str:
    """Map ComfyUI node chatter to friendly stage names."""
    if raw in {"sampling"} or raw.startswith("node"):
        return "generating"
    if raw.startswith("queued"):
        return "queued"
    if raw == "cached":
        return "loading cached graph"
    return raw
