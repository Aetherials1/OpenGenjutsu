# Workflows (engines)

An engine is a folder in `backend/workflows/<id>/` with two files:

```
backend/workflows/wan21-vace/
├── manifest.json   # metadata, defaults, requirements, VRAM gate
└── template.json   # ComfyUI "API format" workflow with @placeholders
```

No Python needed to add, tweak or replace one. Restart the backend, or call
`POST /api/engines/reload`, and it appears in the UI.

The Wan replacement workflows use `sam2.1_hiera_base_plus.safetensors` in the
ComfyUI node. The SAM2 node resolves that selector to its fp16 checkpoint,
`sam2.1_hiera_base_plus-fp16.safetensors`. Do not switch this graph back to a
SAM2 2.0 checkpoint: the video segmentor rejects bounding-box prompts on 2.0,
which breaks source-performer tracking.

## manifest.json

```json
{
  "id": "wan21-vace",
  "label": "Wan 2.1 VACE 1.3B",
  "priority": 10,
  "maturity": "stable",
  "description": "one paragraph shown in the UI",
  "notes": "practical advice shown under the engine picker",
  "defaults": { "width": 832, "height": 480, "fps": 16, "frames": 65,
                "steps": 16, "cfg": 4.0, "shift": 5.0 },
  "max_seconds": 8,
  "min_vram_mb": 7000,
  "required_nodes": ["UNETLoader", "WanVaceToVideo", "KSampler", "..."],
  "required_models": ["diffusion_models/...safetensors"]
}
```

* `priority` — lower is tried first in `auto` mode.
* `min_vram_mb` — hard gate. If the GPU has less, the engine reports
  "not ready" with an explanation instead of OOM-ing mid-render.
* `required_nodes` — must exist in `/object_info`.
* `required_models` — informational; shown as a warning.

## template.json placeholders

Any string value in the graph may contain placeholders:

| placeholder | value | type |
|---|---|---|
| `@char` | staged character image reference | str |
| `@motion` | staged motion video reference | str |
| `@prompt` / `@negative` | text | str |
| `@seed` | resolved seed (never −1) | int |
| `@steps` / `@cfg` / `@shift` | sampler settings | int/float |
| `@width` / `@height` | output size | int |
| `@fps` | frames per second | float |
| `@frames` / `@length` | frame count (already `4n+1`) | int |
| `@duration` | seconds | float |
| `@prefix` / `@job` | output filename prefix / job id | str |

A value that is *exactly* a placeholder keeps its native type (`"@steps"` → the
int `16`). A placeholder inside a longer string is interpolated
(`"og_@seed"` → `"og_1234567"`).

## Making a new template the reliable way

ComfyUI changes; you should not have to read this project's code to fix a
workflow.

1. In ComfyUI's web UI, load the official template for the model
   (`Workflow → Browse Templates → Video → Wan …`).
2. Fill in real values and get it working **in the UI** for a short clip.
3. `Workflow → Export (API)` → save over
   `backend/workflows/<id>/template.json`.
4. Replace four values in that JSON:
   * `LoadImage.inputs.image` → `"@char"`
   * `LoadVideo.inputs.file` → `"@motion"`
   * the positive `CLIPTextEncode.inputs.text` → `"@prompt"`
   * the negative `CLIPTextEncode.inputs.text` → `"@negative"`
   Optionally `KSampler.inputs.seed` → `"@seed"`, `steps` → `"@steps"`,
   `Wan*ToVideo.inputs.width/height/length` → `"@width"/"@height"/"@frames"`.
5. Keep `SaveVideo` (or `SaveAnimatedWEBP`) as the output node — the backend
   reads `outputs[*].images|gifs|videos` from `/history`.
6. Update `required_nodes` in `manifest.json` from the node types you see, then
   click **Self-test** or run `colab_setup.py doctor`.

This is exactly how to recover if a template in this repo drifts from your
ComfyUI version.

## Checking a template without rendering

```bash
python scripts/colab_setup.py doctor
```

```
[BLOCKED] wan22-animate - Wan 2.2 Animate 14B
           ! needs ~21 GB VRAM but this GPU has 15 GB - pick a lighter engine...
[READY  ] wan21-vace - Wan 2.1 VACE 1.3B
```

The same detail appears in the UI under the engine selector.

## Model files

`scripts/download_models.py` resolves globs against the real Hugging Face file
list, downloads into a cache directory (Drive on Colab) and symlinks the result
into `ComfyUI/models/<type>/`.

```bash
python scripts/download_models.py --preset vace --dry-run
python scripts/download_models.py --preset animate-q4
```

Presets: `vace`, `animate` (fp8 14B), `animate-q4` (GGUF), `all`.

If upstream renames a file, `--dry-run` tells you nothing matched; edit the
`SPECS` dict at the top of that script. Symlinks keep Drive usage down and let
you swap quantisations without re-downloading.

## Adding a runpod/A100 profile

Same template, different manifest:

```json
{ "id": "wan22-animate-a100",
  "label": "Wan 2.2 Animate 14B (fp8, 24 GB+)",
  "priority": 1,
  "min_vram_mb": 22000,
  "template": "copy of wan22-animate/template.json" }
```

Copy the folder, change `id`/`label`/`priority`/`min_vram_mb`, and it coexists
with the others.
