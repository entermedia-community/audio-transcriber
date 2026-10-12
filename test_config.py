import os
import torch
from config import Settings

def run_diagnostics():
    print("=" * 60)
    print("         SYSTEM & CONFIGURATION DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Check OS/Environment layers
    print(f"👉 CUDA_VISIBLE_DEVICES (Env): {os.getenv('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    print(f"👉 PyTorch Available GPUs:    {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"👉 Active GPU Name:           {torch.cuda.get_device_name(0)}")
    print("-" * 60)

    # 2. Initialize your Pydantic settings
    settings = Settings()

    print(f"⚙️ Target Device Type:       {settings.device}")
    print(f"⚙️ Target Device Index:      {settings.device_index}")
    print(f"⚙️ Resolved Compute Type:    {settings.compute_type}")
    print(f"⚙️ Whisper Model:            {settings.whisper_model}")
    print("=" * 60)

    # 3. Print VRAM Analysis
    if settings.device == "cuda":
        print("📊 VRAM FOOTPRINT ESTIMATION (large-v3):")
        if settings.compute_type == "float16":
            print("   • Expected Whisper VRAM: ~3.1 GB to 4.0 GB")
            print("   • Status: ⚠️ High risk of OOM when paired with Pyannote on a busy GPU.")
        elif "int8" in settings.compute_type:
            print("   • Expected Whisper VRAM: ~1.5 GB to 1.9 GB")
            print("   • Status: ✅ Optimized! You saved ~50% Whisper VRAM memory usage.")
        print("=" * 60)

if __name__ == "__main__":
    run_diagnostics()