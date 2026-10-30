# Troubleshooting

Start here:

```bash
python scripts/colab_setup.py status     # are the three processes alive?
python scripts/colab_setup.py doctor     # which engine is ready?
python scripts/smoke_test.py             # does the whole pipeline work?
```

Logs live in `/content/ogenjutsu-logs/` on Colab (`logs/` locally):
`comfyui.log`, `backend.log`, `tunnel.log`.

---

## The UI says "offline"

* Check the backend is listening: `curl http://127.0.0.1:7860/api/health`.
* Opening from `file://`? Set the backend URL in ⚙ Settings — a page on disk
  cannot guess where Colab is.
* If the tunnel URL changed (it does on every restart) you must re-open the new
  one; old URLs die with the tunnel.
* `401` / "locked" pill → the token is wrong. Copy `OG_TOKEN` from the notebook
  into Settings.

## "ComfyUI is not reachable at http://127.0.0.1:8188"

`colab_setup.py serve` starts both processes and waits. If ComfyUI dies during
boot (almost always torch/CUDA), `comfyui.log` says why. Common causes:

* runtime type is CPU — switch to `Runtime → Change runtime type → T4 GPU`
* ComfyUI requirements pulled an incompatible torch → `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121`
* out of disk: `df -h /content` — 14B checkpoints are large

## "ComfyUI is missing node(s): WanAnimateToVideo"

Your ComfyUI is older than the model you picked, or the node lives in a custom
node pack.

* `git pull` inside the ComfyUI checkout, restart.
* For GGUF you need <https://github.com/city96/ComfyUI-GGUF>:
  `python scripts/colab_setup.py comfy --with-gguf`
* Otherwise pick another engine in the UI, or build a template from the official
  workflow — see [WORKFLOWS.md](WORKFLOWS.md).

## "WanVaceToVideo (#8) has no input 'strength' in this ComfyUI build"

The template was written for a different ComfyUI version than you have. This
message is intentional and tells you exactly what to change. Either:

* delete that key from `backend/workflows/<id>/template.json`, or
* regenerate the template from the UI (`Export (API)`) as described in
  [WORKFLOWS.md](WORKFLOWS.md).

## CUDA out of memory

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ...
```

* Drop resolution to `832×480` and length to 3–4 s.
* Use `wan21-vace` or `wan22-animate-gguf` instead of the fp8 14B model.
* Start ComfyUI with more aggressive flags:
  `python main.py --lowvram --use-split-cross-attention --reserve-vram 1.0`
* Restart the runtime before a big job — a previous model may still be cached.
  `POST /api/selftest` and cancels can leave weights resident.
* The UI blocks engines whose `min_vram_mb` exceeds your card. If you want to try
  anyway (slow offloading, likely OOM), set `OG_IGNORE_VRAM=1`.

## Renders are very slow

Expected on a free T4. Levers, in order of effect:

1. shorter `seconds` (linear)
2. lower resolution (quadratic-ish)
3. fewer `steps` (linear) — 8–12 steps on VACE is usually enough
4. `cfg` 1.0 with a lightning/distilled LoRA where the engine supports it

Watch `nvidia-smi` while rendering: if utilisation is <50% you are CPU/IO bound
(usually the VAE decode at 720p), not VRAM bound.

## "source has no audio" but the clip clearly has sound

The browser's own detection is a hint only; the backend is authoritative. Check
what ffprobe sees:

```bash
ffprobe -v error -select_streams a -show_entries stream=codec_name -of csv=p=0 clip.mp4
```

Silence → the audio track is not in the container (some screen recorders mux it
separately). Remux with `ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac out.mp4`
and upload that.

If the file *does* have audio but the render is silent, look at the job log line —
it will say `audio: original track attached` or explain why not.

## Audio and video look out of sync

The muxer pads or trims the original track to the render's length (`apad` +
`-shortest`), it never stretches it. If your source was recorded with drift or a
variable frame rate, convert it to constant frame rate first:

```bash
ffmpeg -i in.mp4 -vf fps=30 -c:v libx264 -crf 18 -c:a aac out.mp4
```

## cloudflared prints no URL

* The binary is downloaded to `/usr/local/bin/cloudflared`; check
  `logs/tunnel.log`.
* Corporate/VPN DNS can block `*.trycloudflare.com`. Try a phone hotspot.
* Quick tunnels are rate-limited and occasionally unavailable — wait a minute
  and re-run `tunnel`, or use ngrok (`ngrok http 7860`) and point the UI at that
  URL.
* Never share the URL without `OG_TOKEN` set.

## Colab disconnected

Free runtimes are recycled. Nothing resumes automatically, but:

* models are cached in Drive (`OG_MODELS_DIR`), so `models` is a no-op on the
  next run
* uploaded sources for finished jobs are still in `data/jobs/` — but `/content`
  is gone after a factory reset. Point `OG_DATA_DIR` at Drive if you want
  renders to persist.

Restart: run the notebook cells again. Re-running `system`, `comfy` and `models`
is idempotent.

## Upload returns 413

`OG_MAX_UPLOAD_MB` (default 512). Raise it, but remember the whole file crosses
the tunnel — for anything long, trim it first.

## The mock engine's output makes no sense

It is a placeholder: your character image with the motion clip inset in the
corner, purely to prove upload → queue → progress → audio → download works.
Use a real engine for actual transfer.

## Job stuck at "queued"

Only one job runs at a time. Check `/api/health` → `queue.queued`. If the active
job is hung, `POST /api/jobs/<id>/cancel` interrupts ComfyUI. If the whole
runtime is wedged, restart the runtime.
