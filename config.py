"""
Configuration — all values can be overridden via environment variables.

Example .env:
    WHISPER_MODEL=medium
    DEVICE=cpu
    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _auto_compute_type(device: str) -> str:
    """int8 on CPU, float16 on GPU — best efficiency/quality ratio."""
    return "float16" if device == "cuda" else "int8"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Whisper ────────────────────────────────────────────────────────────
    # "medium" is the sweet spot: fast enough on CPU, excellent accuracy.
    # Use "small" for speed priority, "large-v3" for accuracy priority.
    whisper_model: str = "medium"

    # Auto-detected: cuda if available, else cpu
    device: str = _auto_device()

    # int8 on CPU (≈2× faster, minimal accuracy loss)
    # float16 on GPU (native speed, full precision)
    compute_type: str = ""  # resolved below

    beam_size: int = 5          # 5 = default quality; lower = faster
    whisper_num_workers: int = 1
    whisper_cpu_threads: int = 4

    # ── Pyannote ───────────────────────────────────────────────────────────
    # Required: get a free token at https://huggingface.co/pyannote/speaker-diarization-3.1
    # Accept the model's conditions on the HF page first.
    hf_token: str = os.getenv("HF_TOKEN", "")

    # ── API ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    def model_post_init(self, __context):
        # Resolve compute_type after device is known
        if not self.compute_type:
            object.__setattr__(self, "compute_type", _auto_compute_type(self.device))
