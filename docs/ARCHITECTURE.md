# Architecture

## Processes

Three things run inside the Colab VM (or your machine):

| port | process | job |
|---|---|---|
| 7860 | `uvicorn backend.main:app` | HTTP API, job queue, all ffmpeg work, serves the UI |
| 8188 | ComfyUI | the diffusion pipeline |
| – | `cloudflared` | TCP tunnel: public HTTPS → `127.0.0.1:7860` |

The backend is the only thing exposed. ComfyUI stays on localhost so its
unauthenticated API and file browser are never reachable from the internet.
(`colab_setup.py serve --expose-comfy` exists for debugging, but then you should
tunnel it separately and know what you're doing.)

## Request flow

```
browser
  │  POST /api/jobs  (multipart, motion + character + settings)
  ▼
main.create_job ──► validate with ffprobe
  │                 extract audio to input/audio.wav (if keep_audio)
  │                 reject bad input early (400/413/409)
  ▼
store.create ──► data/jobs/<id>/{job.json,job.log,input/,output/}
  │
  ▼
worker queue (exactly one worker thread = one GPU)
  │
  ├─ 1. prepare inputs           media.prepare_motion()   exact frames + fps + size
  │                              media.prepare_character() RGB PNG, ≤1280 px
  ├─ 2. stage into ComfyUI       copy to ComfyUI/input/ogenjutsu/*  (or HTTP upload)
  ├─ 3. inject template          "@motion" → filename, "@seed" → int, ...
  ├─ 4. POST /prompt             returns prompt_id
  ├─ 5. websocket progress       /ws?clientId=  → progress → SSE-free polling by UI
  ├─ 6. GET /history/<id>        → download output via /view
  ├─ 7. normalise to mp4         media.ensure_mp4()
  ├─ 8. audio                 ►  media.mux_audio() or media.strip_audio()
  └─ 9. thumbnail + meta         ffprobe the final file, write job.json
  ▼
browser polls GET /api/jobs/<id> every ~900 ms → progress bar, logs, result
```

### Why normalise inputs before inference

Wan models want a latent length of `4n+1` frames at a fixed fps. Rather than
hoping the caller's clip matches, the worker **re-times the motion source with
ffmpeg** to exactly `frames` frames at the requested fps and resolution
(looping a short clip, trimming a long one). Every engine then receives a
predictable clip, which removes the most common class of GPU-side failures
("latent length mismatch", black frames at the tail, drift).

### Why the queue is single-worker

A free T4 has 16 GB. Two concurrent Wan renders = guaranteed OOM, and the
second one would also evict the first one's weights. One worker also makes
progress, ETA and cancellation meaningful. Queued jobs are visible with their
position.

## Engine layer

`backend/engines/__init__.py` discovers every folder in `backend/workflows/`
containing `manifest.json` + `template.json` and registers it by `id`.
`mock` is always registered.

`TemplateEngine.run()` is model-agnostic:

1. `stage_input()` each file into ComfyUI
2. `inject(template, placeholder_values(spec))` — recursive substitution where a
   value that is *exactly* a placeholder keeps its native type
3. `submit()` → `wait()` → `fetch_output()` → `ensure_mp4()`

`preflight()` is the safety net. It checks:
* ComfyUI reachable
* every `required_nodes` class exists
* every input key used by the template exists on the installed node
  (`/object_info` is the source of truth, so a ComfyUI upgrade that renames an
  input surfaces as *"WanVaceToVideo (#8) has no input 'strength' in this
  ComfyUI build"* instead of a stack trace)
* `min_vram_mb` against the detected GPU

`resolve_default()` sorts only ready `sam2_mix` replacement workflows by
`manifest.priority` (ascending) and picks the first. Pose-only Animate 2 and the
mock composite are never selected automatically for a replacement request. If
no Mix workflow is ready, the API refuses the request with an actionable install
message instead of reporting a misleading source-like render as successful.

## Audio design

Audio is extracted at **upload** time, not after rendering:

* the toggle can then be flipped on a re-run without touching the source video
  again
* `has_source_audio` is known immediately, so the UI can warn before you wait
  15 minutes for a silent render
* muxing is one `-c:v copy` pass — the generated picture is never re-encoded

Duration policy: `apad` + `-shortest`. If the render is 4.2 s and the clip is
7 s, the audio is cut at 4.2 s with a 0.35 s fade. If the render is longer, the
tail is padded with silence.

## State and recovery

* Each job is a directory with a `job.json` written atomically
  (`job.json.tmp` → replace). No database.
* On boot, jobs left `queued`/`running` are marked `error: backend restarted
  before this job finished` — a render cannot be resumed across a restart, but
  the uploads and metadata survive.
* `work/` (prepared inputs, raw model output) is deleted after each job;
  `input/` and `output/` are kept so you can re-download or re-compare.
* Old finished jobs are pruned to `OG_KEEP_JOBS`.

## Frontend

No bundler, no framework. `index.html` + `app.css` + `app.js` served from the
backend root, so `location.origin` is the API. Opened from disk or another host,
the UI uses a saved backend URL from `localStorage` and talks cross-origin (CORS
is enabled by `OG_CORS`).

Progress is polled rather than streamed over SSE/WebSocket on purpose: polling
survives tunnels that buffer or drop long-lived connections, and 1 request/s is
nothing.

## Failure boundaries

| failure | behaviour |
|---|---|
| ffmpeg missing | `health.ffmpeg.ok = false`, generate returns 409 with the install hint |
| ComfyUI down | engine marked not ready with the URL; `mock` still works |
| template/ComfyUI mismatch | preflight problem naming the node and input |
| OOM during render | ComfyUI reports `execution_error`; job marked `error` with the message |
| tunnel drops | UI shows "lost contact"; job keeps running server-side |
| runtime recycle | remaining jobs marked interrupted; models cached on Drive |
