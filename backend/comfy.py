"""Minimal, dependency-light client for the ComfyUI HTTP + websocket API.

The backend talks to ComfyUI on localhost (both live in the same Colab VM), so
inputs are staged by copying files into ComfyUI's `input` directory when that
path is known - far faster and more reliable than HTTP uploads for video.

Progress is streamed from ComfyUI's websocket so the UI can show a real
percentage instead of a spinner.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from .config import settings

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


class ComfyError(RuntimeError):
    pass


def gpu_info() -> dict[str, Any]:
    """Best-effort GPU description via nvidia-smi."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        line = out.stdout.decode("utf-8", "replace").strip().splitlines()
        if out.returncode == 0 and line:
            name, total, used, driver = [p.strip() for p in line[0].split(",")]
            return {
                "available": True,
                "name": name,
                "vram_total_mb": int(float(total)),
                "vram_used_mb": int(float(used)),
                "driver": driver,
            }
    except Exception:
        pass
    return {"available": False, "name": None, "vram_total_mb": 0, "vram_used_mb": 0}


class ComfyClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base = (base_url or settings.comfy_url).rstrip("/")
        parsed = urlparse(self.base)
        self.ws_base = (
            ("wss" if parsed.scheme == "https" else "ws")
            + "://"
            + parsed.netloc
            + (parsed.path or "")
        )

    # ------------------------------------------------------------------ state
    def ping(self, timeout: float = 2.0) -> bool:
        try:
            r = requests.get(f"{self.base}/system_stats", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def system_stats(self) -> dict[str, Any] | None:
        try:
            r = requests.get(f"{self.base}/system_stats", timeout=6)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            return None

    def object_info(self) -> dict[str, Any]:
        try:
            r = requests.get(f"{self.base}/object_info", timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            raise ComfyError(f"cannot read ComfyUI node info: {exc}") from exc

    def node_names(self) -> set[str]:
        try:
            return set(self.object_info().keys())
        except ComfyError:
            return set()

    # ----------------------------------------------------------------- inputs
    def stage_input(self, local: Path, name: str) -> str:
        """Make ``local`` available to ComfyUI and return the name to reference it.

        Files are placed *directly* in ComfyUI's ``input`` directory with an
        ``og_`` prefix: several core loader nodes call ``os.path.basename`` on
        the value, so a subfolder-qualified name would break them. The prefix
        also makes cleanup unambiguous.
        """
        safe = "og_" + _safe_name(name)
        if settings.comfy_input_dir:
            target_dir = Path(settings.comfy_input_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, target_dir / safe)
            return safe

        mime = "video/mp4" if safe.lower().endswith((".mp4", ".webm", ".mov")) else None
        try:
            with open(local, "rb") as handle:
                files = {"image": (safe, handle, mime or "application/octet-stream")}
                data = {"overwrite": "true", "subfolder": "", "type": "input"}
                r = requests.post(
                    f"{self.base}/upload/image", files=files, data=data, timeout=600
                )
            r.raise_for_status()
            payload = r.json()
        except (requests.RequestException, ValueError) as exc:
            raise ComfyError(f"upload to ComfyUI failed: {exc}") from exc

        subfolder = payload.get("subfolder") or ""
        if subfolder:
            # ComfyUI stored it in a subfolder anyway; reference it with the
            # annotation form some nodes understand.
            return f"{subfolder}/{payload['name']}"
        return payload["name"]

    def clear_staging(self) -> None:
        """Delete inputs this backend staged (never touches the user's files)."""
        if not settings.comfy_input_dir:
            return
        root = Path(settings.comfy_input_dir)
        for path in root.glob("og_*"):
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
        shutil.rmtree(root / "ogenjutsu", ignore_errors=True)

    # ------------------------------------------------------------------- jobs
    def submit(self, graph: dict[str, Any], client_id: str) -> str:
        try:
            r = requests.post(
                f"{self.base}/prompt",
                json={"prompt": graph, "client_id": client_id},
                timeout=60,
            )
        except requests.RequestException as exc:
            raise ComfyError(f"cannot reach ComfyUI: {exc}") from exc

        if r.status_code != 200:
            raise ComfyError(_format_prompt_error(r))
        payload = r.json()
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"ComfyUI returned no prompt id: {payload}")
        return prompt_id

    def wait(
        self,
        prompt_id: str,
        client_id: str,
        on_progress: ProgressFn | None = None,
        should_cancel: CancelFn | None = None,
        timeout: float = 3600,
    ) -> dict[str, Any]:
        """Block until the prompt finishes. Returns the history entry."""
        deadline = time.time() + timeout
        seen_nodes: dict[str, str] = {}

        try:
            import websocket  # type: ignore
        except Exception:
            return self._wait_polling(prompt_id, on_progress, should_cancel, deadline)

        url = f"{self.ws_base}/ws?clientId={client_id}"
        try:
            ws = websocket.create_connection(url, timeout=15)
        except Exception:
            return self._wait_polling(prompt_id, on_progress, should_cancel, deadline)

        try:
            ws.settimeout(1.0)
            while True:
                if time.time() > deadline:
                    raise ComfyError("generation timed out")
                if should_cancel and should_cancel():
                    self.cancel(prompt_id)
                    raise ComfyError("__canceled__")

                try:
                    raw = ws.recv()
                except Exception as exc:
                    if type(exc).__name__ in {"WebSocketTimeoutException", "TimeoutError"}:
                        continue
                    raise ComfyError(f"websocket closed: {exc}") from exc

                if not raw or isinstance(raw, bytes):
                    continue  # binary frames = preview images
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue

                kind = msg.get("type")
                data = msg.get("data") or {}

                if kind == "progress":
                    value = float(data.get("value") or 0)
                    total = float(data.get("max") or 0) or 1.0
                    if data.get("prompt_id") not in (None, prompt_id):
                        continue
                    if on_progress:
                        on_progress(min(0.97, 0.05 + 0.92 * (value / total)), "sampling")
                elif kind == "executing":
                    node = data.get("node")
                    pid = data.get("prompt_id")
                    if pid != prompt_id:
                        continue
                    if node is None:
                        break
                    seen_nodes[str(node)] = str(node)
                    if on_progress:
                        on_progress(0.05, f"node {node}")
                elif kind == "execution_error":
                    if data.get("prompt_id") not in (None, prompt_id):
                        continue
                    detail = (
                        f"{data.get('node_type', 'node')}: {data.get('exception_message', '')}"
                        .strip()
                    )
                    raise ComfyError(f"ComfyUI execution error - {detail}")
                elif kind == "execution_cached":
                    if on_progress:
                        on_progress(0.06, "cached")
                elif kind == "status":
                    queue = (
                        (data.get("status") or {}).get("exec_info", {}).get("queue_remaining")
                    )
                    if queue is not None and on_progress:
                        on_progress(0.03, f"queued ({queue})")
        finally:
            try:
                ws.close()
            except Exception:
                pass

        return self.history(prompt_id)

    def _wait_polling(
        self,
        prompt_id: str,
        on_progress: ProgressFn | None,
        should_cancel: CancelFn | None,
        deadline: float,
    ) -> dict[str, Any]:
        started = time.time()
        while time.time() < deadline:
            if should_cancel and should_cancel():
                self.cancel(prompt_id)
                raise ComfyError("__canceled__")
            entry = self.history(prompt_id)
            if entry:
                return entry
            if on_progress:
                elapsed = time.time() - started
                on_progress(min(0.9, 0.05 + elapsed / 400.0), "generating")
            time.sleep(3)
        raise ComfyError("generation timed out")

    def history(self, prompt_id: str) -> dict[str, Any]:
        try:
            r = requests.get(f"{self.base}/history/{prompt_id}", timeout=30)
            r.raise_for_status()
            payload = r.json() or {}
        except (requests.RequestException, ValueError) as exc:
            raise ComfyError(f"cannot read ComfyUI history: {exc}") from exc
        entry = payload.get(prompt_id)
        if not entry:
            return {}
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            raise ComfyError(_format_status_error(status))
        return entry

    def cancel(self, prompt_id: str) -> None:
        """Remove from queue if still queued, then interrupt whatever is running."""
        try:
            requests.post(
                f"{self.base}/queue", json={"delete": [prompt_id]}, timeout=10
            )
        except requests.RequestException:
            pass
        try:
            requests.post(f"{self.base}/interrupt", json={}, timeout=10)
        except requests.RequestException:
            pass

    # --------------------------------------------------------------- outputs
    def fetch_output(self, entry: dict[str, Any], dest_dir: Path) -> Path:
        """Download the primary output file described by a history entry.

        The remote filename (and therefore its extension) is preserved inside
        ``dest_dir`` so callers can tell an mp4 from an animated webp.
        """
        outputs = entry.get("outputs") or {}
        dest_dir.mkdir(parents=True, exist_ok=True)
        for node_output in outputs.values():
            for key in ("videos", "gifs", "images", "audio"):
                items = node_output.get(key) or []
                for item in reversed(items):
                    if not isinstance(item, dict) or "filename" not in item:
                        continue
                    dest = dest_dir / _safe_name(Path(item["filename"]).name)
                    saved = self._download(
                        item["filename"],
                        item.get("subfolder", ""),
                        item.get("type", "output"),
                        dest,
                    )
                    if saved:
                        return saved
        raise ComfyError("ComfyUI finished but produced no output file")

    def _download(
        self, filename: str, subfolder: str, kind: str, dest: Path
    ) -> Path | None:
        params = {"filename": filename, "subfolder": subfolder, "type": kind}
        try:
            with requests.get(
                f"{self.base}/view", params=params, stream=True, timeout=300
            ) as r:
                if r.status_code != 200:
                    return None
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as handle:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
        except requests.RequestException:
            return None
        return dest if dest.exists() and dest.stat().st_size else None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def new_client_id() -> str:
    return uuid.uuid4().hex


def _safe_name(name: str) -> str:
    keep = "".join(c for c in name if c.isalnum() or c in "-_. ")
    return (keep.strip().replace(" ", "_") or "input")[:120]


def _format_prompt_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"ComfyUI rejected the workflow ({response.status_code})"

    parts: list[str] = []
    error = payload.get("error")
    if isinstance(error, dict):
        parts.append(str(error.get("message") or error.get("type") or error))
    elif error:
        parts.append(str(error))

    for node_id, info in (payload.get("node_errors") or {}).items():
        if not isinstance(info, dict):
            continue
        label = (info.get("class_type") or "node") + f" #{node_id}"
        for sub in info.get("errors") or []:
            parts.append(f"{label}: {sub.get('message', '')} {sub.get('details', '')}".strip())

    if not parts:
        parts.append(f"ComfyUI rejected the workflow ({response.status_code})")
    return "Workflow invalid - " + " | ".join(parts)


def _format_status_error(status: dict[str, Any]) -> str:
    messages: list[str] = []
    for sub in status.get("messages") or []:
        if not isinstance(sub, (list, tuple)) or len(sub) < 2:
            continue
        name, payload = sub[0], sub[1]
        if name == "execution_error" and isinstance(payload, dict):
            messages.append(
                f"{payload.get('node_type', 'node')}: {payload.get('exception_message', '')}"
            )
    return "ComfyUI execution failed - " + ("; ".join(messages) or "unknown error")
