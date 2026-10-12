"""
Configuration — all values can be overridden via environment variables.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Define the absolute target to your shared file explicitly
_HOME_ENV = Path.home() / ".env"

# Force Python to read it instantly and merge it with os.environ system-wide
if _HOME_ENV.exists():
    load_dotenv(dotenv_path=_HOME_ENV)


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

class Settings(BaseSettings):
   # This reads straight from the system environment we just injected above
    model_config = SettingsConfigDict(extra="ignore")

    # ── CUDA Environmental Mapping ──────────────────────────────────────────
    # Now it dynamically catches the value from your home .env perfectly
    cuda_visible_devices: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")

    # ── Whisper ────────────────────────────────────────────────────────────
    whisper_model: str = os.getenv("WHISPER_MODEL", "large-v3")
    device: str = _auto_device()
    device_index: int = 0  
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

       # 3. Resolve precision dynamically
        if not self.compute_type:
            if self.device == "cuda":
                env_cuda_type = os.getenv("CUDA_COMPUTE_TYPE") or os.getenv("CUDA_COPUTE_TYPE")
                # CHANGED: 'int8_float16' is the correct supported string name
                resolved_type = env_cuda_type if env_cuda_type else "int8_float16"
            else:
                resolved_type = "int8"
                
            object.__setattr__(self, "compute_type", resolved_type)