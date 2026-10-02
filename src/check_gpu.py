"""Check PyTorch CUDA and faster-whisper/CTranslate2 GPU inference."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
from config import HF_CACHE_DIR

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)


def main():
    import numpy as np
    import torch

    print(f"PyTorch version: {torch.__version__}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")
    if available:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(f"GPU name: {props.name}")
        print(f"Compute capability: {props.major}.{props.minor}")
        print(f"Free VRAM: {free_bytes / (1024 ** 3):.2f} GiB / {total_bytes / (1024 ** 3):.2f} GiB")
    else:
        print("GPU name: unavailable")
        print("Compute capability: unavailable")
        print("Free VRAM: unavailable")

    print("\nTesting faster-whisper (CTranslate2) CUDA inference with the tiny model...")
    if not available:
        print("FAIL: PyTorch cannot access CUDA; CTranslate2 GPU test skipped.")
        return 1

    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        audio = np.zeros(2 * 16000, dtype=np.float32)
        segments, info = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        list(segments)
        print(f"PASS: CTranslate2 loaded tiny on CUDA and completed inference (detected language: {info.language}).")
        return 0
    except Exception as exc:
        print("FAIL: CTranslate2 could not load or run the tiny model on CUDA.")
        print(f"Reason: {type(exc).__name__}: {exc}")
        print("Do not assume faster-whisper GPU transcription will work; the transformers CUDA backend may be needed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
