#!/usr/bin/env python
"""A tiny stand-in for ComfyUI - development and demo use only.

It implements just enough of the ComfyUI API for OpenGenjutsu to drive it
end-to-end *without a GPU*: uploads, /prompt, websocket progress, /history,
/view. Instead of running a diffusion model it renders a test clip with ffmpeg,
so the real ``TemplateEngine`` code path (input staging, placeholder injection,
progress plumbing, output download, mp4 normalisation) is exercised exactly as
it will be on Colab.

    python scripts/fake_comfy.py --port 8188
    # in another shell
    python run.py

Deliberately NOT a ComfyUI emulator: no workflow validation beyond what is needed
to look successful, no model loading, no correctness guarantees.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = ROOT / "backend" / "workflows"

STATE_DIR = ROOT / "data" / "fake_comfy"
INPUT_DIR = STATE_DIR / "input"
OUTPUT_DIR = STATE_DIR / "output"

app = FastAPI(title="fake-comfyui")

SOCKETS: list[WebSocket] = []
JOBS: dict[str, dict] = {}
RUNNING: dict[str, threading.Thread] = {}
INTERRUPTED: set[str] = set()


# --------------------------------------------------------------------------- #
# node catalogue derived from the workflow templates
# --------------------------------------------------------------------------- #


def node_catalogue() -> dict:
    """Every class_type used by any template, with the union of inputs it uses."""
    catalogue: dict[str, dict] = {}
    for template in WORKFLOW_DIR.glob("*/template.json"):
        try:
            graph = json.loads(template.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            ctype = node.get("class_type")
            if not ctype:
                continue
            entry = catalogue.setdefault(ctype, {"input": {"required": {}, "optional": {}}})
            for key, value in (node.get("inputs") or {}).items():
                spec = "INT" if isinstance(value, int) and not isinstance(value, bool) else "STRING"
                if isinstance(value, float):
                    spec = "FLOAT"
                elif isinstance(value, list):
                    spec = "CONNECTION"
                elif isinstance(value, dict):
                    continue
                entry["input"]["optional"][key] = [spec, {}]
    # nodes the frontends always expect to exist
    for name in ("CheckpointLoaderSimple", "LoadImage", "LoadVideo", "SaveVideo"):
        catalogue.setdefault(name, {"input": {"required": {}, "optional": {}}})
    return catalogue


CATALOGUE = node_catalogue()


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #


async def broadcast(message: dict) -> None:
    dead = []
    for socket in SOCKETS:
        try:
            await socket.send_text(json.dumps(message))
        except Exception:
            dead.append(socket)
    for socket in dead:
        if socket in SOCKETS:
            SOCKETS.remove(socket)


def guess_settings(graph: dict) -> dict:
    """Pull width/height/length/fps out of the submitted graph, best effort."""
    found = {"width": 832, "height": 480, "frames": 49, "fps": 16}
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key in ("width", "height"):
            if isinstance(inputs.get(key), int):
                found[key] = inputs[key]
        for key in ("length", "frame_load_cap", "frames"):
            if isinstance(inputs.get(key), int) and inputs[key] > 1:
                found["frames"] = inputs[key]
        if isinstance(inputs.get("fps"), (int, float)) and inputs["fps"] > 0:
            found["fps"] = float(inputs["fps"])
    return found


def render(prompt_id: str, graph: dict) -> None:
    import asyncio
    import sys

    sys.path.insert(0, str(ROOT))
    from backend import media

    job = JOBS[prompt_id]
    settings = guess_settings(graph)
    steps = 12
    frames = max(5, min(settings["frames"], 121))
    fps = max(1.0, min(settings["fps"], 30.0))
    seconds = frames / fps

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            broadcast({"type": "execution_start", "data": {"prompt_id": prompt_id}})
        )

        for name, node_id in (("model", "1"), ("sampler", "10")):
            if prompt_id in INTERRUPTED:
                raise RuntimeError("interrupted")
            loop.run_until_complete(
                broadcast(
                    {
                        "type": "executing",
                        "data": {"node": node_id, "prompt_id": prompt_id, "label": name},
                    }
                )
            )
            time.sleep(0.3)

        for step in range(1, steps + 1):
            if prompt_id in INTERRUPTED:
                raise RuntimeError("interrupted")
            time.sleep(0.28)
            loop.run_until_complete(
                broadcast(
                    {
                        "type": "progress",
                        "data": {
                            "value": step,
                            "max": steps,
                            "prompt_id": prompt_id,
                            "node": "10",
                        },
                    }
                )
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"ogenjutsu_{prompt_id[:8]}_00001.mp4"
        target = OUTPUT_DIR / filename
        media.make_test_clip(
            target,
            seconds=seconds,
            width=max(64, settings["width"] - settings["width"] % 2),
            height=max(64, settings["height"] - settings["height"] % 2),
            fps=fps,
        )

        job.update(
            finished=True,
            outputs={
                "14": {
                    "images": [
                        {"filename": filename, "subfolder": "", "type": "output"}
                    ]
                }
            },
        )
        loop.run_until_complete(
            broadcast({"type": "executing", "data": {"node": None, "prompt_id": prompt_id}})
        )
        loop.close()
    except Exception as exc:  # noqa: BLE001
        job.update(finished=True, error=str(exc), outputs={})
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(
                broadcast(
                    {
                        "type": "execution_error",
                        "data": {
                            "prompt_id": prompt_id,
                            "node_type": "KSampler",
                            "exception_message": str(exc),
                        },
                    }
                )
            )
            loop.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #


@app.get("/system_stats")
def system_stats() -> dict:
    return {
        "system": {"os": "fake", "comfyui_version": "fake-1.0", "python_version": "3.12"},
        "devices": [
            {
                "name": "Fake GPU (ffmpeg)",
                "type": "cuda",
                "index": 0,
                "vram_total": 16 * 1024**3,
                "vram_free": 15 * 1024**3,
            }
        ],
    }


@app.get("/object_info")
def object_info() -> dict:
    return CATALOGUE


@app.post("/upload/image")
async def upload_image(
    image: UploadFile = File(...),
    overwrite: str = Form("true"),
    subfolder: str = Form(""),
    type: str = Form("input"),
) -> dict:
    folder = INPUT_DIR / subfolder if subfolder else INPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    name = Path(image.filename or "upload.bin").name
    with open(folder / name, "wb") as handle:
        shutil.copyfileobj(image.file, handle)
    return {"name": name, "subfolder": subfolder, "type": type}


@app.post("/prompt")
async def submit(request: Request) -> JSONResponse:
    payload = await request.json()
    graph = payload.get("prompt") or {}
    if not isinstance(graph, dict) or not graph:
        return JSONResponse({"error": {"message": "empty workflow"}}, status_code=400)

    prompt_id = uuid.uuid4().hex
    JOBS[prompt_id] = {"finished": False, "outputs": {}, "graph": graph}
    thread = threading.Thread(target=render, args=(prompt_id, graph), daemon=True)
    RUNNING[prompt_id] = thread
    thread.start()
    await broadcast({"type": "status", "data": {"prompt_id": prompt_id}})
    return JSONResponse({"prompt_id": prompt_id, "number": 0})


@app.get("/history/{prompt_id}")
def history(prompt_id: str) -> dict:
    job = JOBS.get(prompt_id)
    if not job or not job.get("finished"):
        return {}
    if job.get("error") and job["error"] != "interrupted":
        return {
            prompt_id: {
                "status": {
                    "status_str": "error",
                    "messages": [
                        ["execution_error", {"node_type": "KSampler", "exception_message": job["error"]}]
                    ],
                },
                "outputs": {},
            }
        }
    return {prompt_id: {"status": {"status_str": "success"}, "outputs": job["outputs"]}}


@app.get("/view")
def view(filename: str, subfolder: str = "", type: str = "output") -> FileResponse:
    path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


@app.post("/queue")
async def queue(request: Request) -> dict:
    try:
        payload = await request.json()
    except ValueError:
        payload = {}
    for prompt_id in payload.get("delete") or []:
        INTERRUPTED.add(prompt_id)
    return {"ok": True}


@app.post("/interrupt")
async def interrupt() -> dict:
    for prompt_id, job in JOBS.items():
        if not job.get("finished"):
            INTERRUPTED.add(prompt_id)
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(socket: WebSocket) -> None:
    await socket.accept()
    SOCKETS.append(socket)
    await socket.send_text(json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}}}))
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if socket in SOCKETS:
            SOCKETS.remove(socket)


def main() -> None:
    parser = argparse.ArgumentParser(description="fake ComfyUI for OpenGenjutsu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    args = parser.parse_args()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"fake ComfyUI on http://{args.host}:{args.port} | nodes: {len(CATALOGUE)}")
    print("state:", STATE_DIR)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
