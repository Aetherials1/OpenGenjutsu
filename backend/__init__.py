"""OpenGenjutsu - browser-driven motion transfer on a Colab GPU.

Package layout:
    config   settings from environment
    media    ffmpeg/ffprobe: probe, normalise inputs, mux the original audio
    comfy    ComfyUI HTTP + websocket client
    engines  pluggable render backends (workflow templates + mock)
    store    job persistence
    worker   single-GPU job queue
    main     FastAPI app / web UI host
"""

from .config import VERSION as __version__

__all__ = ["__version__"]
