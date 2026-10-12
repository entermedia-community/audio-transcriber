"""
Configuration — all values can be overridden via environment variables.
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
    # This automatically pulls ANY variable matching these names from your root .env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── CUDA Environmental Mapping ──────────────────────────────────────────
    # Reads CUDA_VISIBLE_DEVICES from your root .env if it is declared there
    cuda_visible_devices: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")

    # ── Whisper ────────────────────────────────────────────────────────────
    whisper_model: str = "large-v3"
    device: str = _auto_device()
    device_index: int = 0  # Automatically falls to 0 to align with masked visible arrays
    compute_type: str = ""  

    beam_size: int = 5          
    whisper_num_workers: int = 1
    whisper_cpu_threads: int = 4

    # ── Pyannote ───────────────────────────────────────────────────────────
    hf_token: str = os.getenv("HF_TOKEN", "")

    # ── API ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 5600

    def model_post_init(self, __context):
        # 1. Standardize string variations (e.g., if DEVICE="cuda:0" or "cuda:1")
        if ":" in self.device:
            dev_type, idx = self.device.split(":")
            object.__setattr__(self, "device", dev_type.strip())
            try:
                object.__setattr__(self, "device_index", int(idx))
            except ValueError:
                pass

        # 2. Safety Override: If CUDA_VISIBLE_DEVICES isolates a single GPU, 
        # any index higher than 0 is invisible to the runtime environment.
        if "," not in self.cuda_visible_devices and self.cuda_visible_devices.isdigit():
            # If it's a single assigned card like "0", "1", or "2", the OS maps it as index 0 inside Python
            object.__setattr__(self, "device_index", 0)

        # 3. Resolve precision
        if not self.compute_type:
            object.__setattr__(self, "compute_type", _auto_compute_type(self.device))