# OpenGenjutsu

Browser-driven **motion transfer** that runs on your own GPU — a Google Colab free
tier T4, or your local card. Upload a **motion clip** and a **character image**,
get back a video of your character performing that motion. A toggle decides
whether the original clip's **audio** is carried into the finished render.

```
   your browser                      Google Colab (free T4)                   disk
 ┌──────────────┐   HTTPS tunnel   ┌──────────────────────────────┐
 │  web UI      │ ───────────────▶ │ FastAPI backend   :7860      │──▶ data/jobs/<id>/
 │  (no build)  │                  │   queue → ffmpeg → audio mux │
 │  or local    │ ◀─── mp4 ─────── │ ComfyUI engine    :8188      │
 └──────────────┘                  └──────────────────────────────┘
                                        ↑ models cached on Drive
```

The UI is plain HTML/CSS/JS served by the backend, so it works three ways:
served by Colab through the tunnel, opened locally against the Colab backend, or
run entirely on your machine with a local GPU.

---

## What it does (and what it doesn't)

| | |
|---|---|
| **In** | motion video (the performance to copy, any length/fps/size) + character image |
| **Out** | mp4 at 480p/720p, 1–8 s, 16 fps, character performing the motion |
| **Audio** | toggle: mux the motion clip's original track (`aac 192k`, trimmed/padded to the render's exact length, gentle fade-out) or leave the render silent |
| **Engines** | Wan 2.2 Animate 14B (fp8 / GGUF Q4), Wan 2.1 VACE 1.3B, and a GPU-free mock |

Two things worth being straight about:

* **"Perfect" is not a real guarantee.** No open model reproduces a performance
  frame-for-frame. Wan Animate is the closest thing available and keeps identity
  and body motion well; expect small face/hand artefacts on fast movement, and
  expect the background to be redrawn rather than preserved.
* **Free T4 is 16 GB and time-limited.** A 4 s 480p clip takes roughly 2–6 min on
  VACE 1.3B and 10–25 min on 14B GGUF Q4. Colab will drop the runtime after a
  few idle hours; your models are cached on Drive so restarts are quick.

---

## Quick start — Google Colab (free GPU)

1. Put this folder in a GitHub repo (or zip it and upload to `/content`).
2. Open [`notebooks/OpenGenjutsu_Colab.ipynb`](notebooks/OpenGenjutsu_Colab.ipynb)
   in Colab, set `Runtime → Change runtime type → T4 GPU`.
3. Edit the first cell (`REPO_URL` or use the zip upload path, `TOKEN`, `PRESET`)
   and run the cells top to bottom.
4. The last cell prints a `https://….trycloudflare.com` URL and your access token.
   Open it on any device — phone included.

On a T4 the notebook picks these by default:

| | |
|---|---|
| engine | `wan22-animate-gguf` if ComfyUI-GGUF is installed, else `wan21-vace` |
| models | `--preset vace` (≈11 GB) or `--preset animate-q4` (≈14 GB, best quality) |
| ComfyUI flags | `--lowvram --use-split-cross-attention --preview-method none` |

Run `!python scripts/colab_setup.py doctor` any time to see exactly which engine is
ready and what is missing.

---

## Quick start — local

```powershell
# 1. backend deps
python -m pip install -r requirements.txt

# 2. run it (mock engine: no GPU needed, proves the whole pipeline)
python run.py
#    -> http://127.0.0.1:7860
```

With a local GPU and ComfyUI installed:

```powershell
# ComfyUI on :8188 (adjust the path to your checkout)
python ..\ComfyUI\main.py --listen 127.0.0.1 --port 8188

# point the backend at it
$env:OG_COMFY_INPUT_DIR = "C:\path\to\ComfyUI\input"
python run.py
```

Then click **Self-test** in the UI. It renders a synthetic clip with a tone and
verifies upload → queue → render → audio mux → download without you touching
anything. From a shell: `python scripts/smoke_test.py`.

### Developing without a GPU

Two helpers make the whole app testable on any laptop:

```bash
python tests/test_engine.py        # 40+ structural checks: templates, injection, frames
python scripts/fake_comfy.py       # a stand-in ComfyUI on :8188 (ffmpeg, no GPU)
python scripts/smoke_test.py --engine wan21-vace   # real TemplateEngine path
```

`fake_comfy.py` implements `/system_stats`, `/object_info`, `/prompt`, `/ws`,
`/history` and `/view`, so the actual ComfyUI client, placeholder injection,
websocket progress and output retrieval run unmodified. It renders a test pattern
instead of a diffusion result — enough to build and debug the UI, an engine
template, or the audio toggle before you burn Colab minutes.

---

## The audio toggle

This is the part people usually get wrong, so it is handled explicitly:

1. On upload the motion clip's first audio stream is extracted once to
   `data/jobs/<id>/input/audio.wav` (48 kHz stereo PCM).
2. After the render is converted to H.264 mp4, if the toggle is **on**:
   `ffmpeg -i render.mp4 -i audio.wav -filter_complex "[1:a]volume=…,afade=t=out:…,apad[a]" -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -shortest`
   * `apad` + `-shortest` → audio is padded with silence when the render is
     longer, and cut when it is shorter, so the file always ends exactly with
     the video. No trailing silence, no truncated word mid-scream.
   * video is stream-copied, so muxing never re-encodes or degrades the picture.
3. Toggle **off** → the audio stream is dropped (`-an`) and the file is silent.
4. A source with no audio stream is detected at upload; the UI warns you and the
   render is silent.

`audio_volume` (0–1.5) applies linear gain. `0` behaves like "off".

---

## How it is built

| Path | Role |
|---|---|
| `backend/main.py` | FastAPI: uploads, job API, auth, serves the UI |
| `backend/media.py` | ffprobe/ffmpeg: probe, re-time inputs, **audio mux**, thumbnails |
| `backend/comfy.py` | ComfyUI client: input staging, queueing, live websocket progress |
| `backend/engines/` | engine registry + `TemplateEngine` + `MockEngine` |
| `backend/workflows/` | one folder per engine: `manifest.json` + `template.json` |
| `backend/worker.py` | one job at a time (one GPU), cancel, progress, finalise |
| `backend/store.py` | per-job directory + JSON state, survives restarts |
| `frontend/` | no-build UI |
| `notebooks/` | Colab notebook (generated by `build_notebook.py`) |
| `scripts/` | Colab bootstrap, model downloader, smoke test |

Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Engines

| id | model | VRAM | speed (4 s, 480p, T4) | status |
|---|---|---|---|---|
| `wan22-animate-gguf` | Wan 2.2 Animate 14B Q4_K_M | ~14 GB | 10–25 min | experimental, best quality |
| `wan22-animate` | Wan 2.2 Animate 14B fp8 | ~22 GB | 8–18 min | experimental, needs ≥22 GB |
| `wan21-vace` | Wan 2.1 VACE 1.3B fp16 | ~7 GB | 2–6 min | stable, good default |
| `mock` | none (ffmpeg) | 0 | 3 s | test only |

Selection is automatic: the backend probes ComfyUI's installed nodes, the
available VRAM, and each template's declared inputs, then picks the highest
priority engine that passes. You can force one in the UI or with `OG_ENGINE`.

Adding or fixing an engine is a data change — see
[`docs/WORKFLOWS.md`](docs/WORKFLOWS.md). Templates are validated against
`/object_info` before every render, so a ComfyUI version bump shows up as a
readable message instead of an obscure failure.

---

## API

All endpoints are under `/api`. If `OG_TOKEN` is set, send `X-Auth-Token: <token>`
(or `?token=` for media URLs that a `<video>` element loads).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | version, GPU, ComfyUI, ffmpeg, queue, engine readiness |
| `GET` | `/engines` | engine cards + defaults |
| `POST` | `/jobs` | multipart: `motion`, `character` + settings → job |
| `GET` | `/jobs?limit=30` | recent jobs |
| `GET` | `/jobs/{id}` | status, progress, stage, logs |
| `GET` | `/jobs/{id}/result` | the mp4 (`?download=1` to attach) |
| `GET` | `/jobs/{id}/source` | the uploaded motion clip (for compare) |
| `GET` | `/jobs/{id}/character` | the reference image |
| `GET` | `/jobs/{id}/thumb` | preview jpeg |
| `POST` | `/jobs/{id}/cancel` | cancel queued/running (interrupts ComfyUI) |
| `DELETE` | `/jobs/{id}` | delete job + files |
| `POST` | `/selftest` | render a synthetic clip end-to-end |

`POST /jobs` fields: `engine, prompt, negative, width, height, fps, seconds,
steps, cfg, seed, keep_audio, audio_volume`.

```bash
curl -X POST "$URL/api/jobs" -H "X-Auth-Token: $TOKEN" \
  -F motion=@dance.mp4 -F character=@me.png \
  -F seconds=4 -F width=832 -F height=480 -F keep_audio=true
```

---

## Configuration

Set as environment variables or in a `.env` next to `run.py`
(see [`.env.example`](.env.example)).

| var | default | |
|---|---|---|
| `OG_TOKEN` | empty | shared secret; **set it whenever you expose a tunnel** |
| `OG_COMFY_URL` | `http://127.0.0.1:8188` | |
| `OG_COMFY_INPUT_DIR` | empty | ComfyUI `input/`; enables fast file staging |
| `OG_ENGINE` | `auto` | force an engine id |
| `OG_DATA_DIR` | `./data` | uploads + renders |
| `OG_KEEP_JOBS` | `40` | history retained on disk |
| `OG_MAX_UPLOAD_MB` | `512` | |
| `OG_CORS` | `*` | allowed browser origins |

Security note: a `trycloudflare.com` URL is public. With `OG_TOKEN` unset anyone
with the link can spend your GPU quota. The notebook generates a random token for
you — keep it.

---

## Troubleshooting

[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) covers the hits: CUDA OOM,
ComfyUI node/template mismatches, cloudflared not printing a URL, "source has no
audio", slow renders, and Colab disconnects.

Quick diagnosis:

```bash
python scripts/colab_setup.py doctor     # engine readiness
python scripts/smoke_test.py             # full pipeline
python scripts/colab_setup.py status     # processes + health
```

---

## Licence / credits

Your code, your GPU. The models it drives are third-party and carry their own
licences: [Wan 2.1/2.2](https://github.com/Wan-Video/Wan2.1),
[ComfyUI](https://github.com/comfyanonymous/ComfyUI),
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF). Check them before
commercial use of generated footage.
