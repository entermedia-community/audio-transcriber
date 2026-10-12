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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Whisper ────────────────────────────────────────────────────────────
    whisper_model: str = "large-v3"
    device: str = _auto_device()
    device_index: int = 0  

    # Automatically tries to map from your .env if COMPUTE_TYPE is declared there.
    # Otherwise, we resolve it dynamically below using your custom variable rules.
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
        # 1. Normalize device strings (e.g., if DEVICE="cuda:0")
        if ":" in self.device:
            dev_type, idx = self.device.split(":")
            object.__setattr__(self, "device", dev_type.strip())
            try:
                object.__setattr__(self, "device_index", int(idx))
            except ValueError:
                pass

        # 2. Safety Override: Align with single CUDA_VISIBLE_DEVICES mappings
        cuda_env = os.getenv("CUDA_VISIBLE_DEVICES", "0")
        if "," not in cuda_env and cuda_env.isdigit():
            object.__setattr__(self, "device_index", 0)

        # 3. Resolve precision based on CUDA_COMPUTE_TYPE environment fallback
        if not self.compute_type:
            if self.device == "cuda":
                # Check for CUDA_COMPUTE_TYPE (handling potential typo variations too)
                env_cuda_type = os.getenv("CUDA_COMPUTE_TYPE") or os.getenv("CUDA_COPUTE_TYPE")
                resolved_type = env_cuda_type if env_cuda_type else "float16_int8"
            else:
                # Standard CPU precision fallback
                resolved_type = "int8"
                
            object.__setattr__(self, "compute_type", resolved_type)