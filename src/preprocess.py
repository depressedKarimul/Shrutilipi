"""Clean a source recording into mono 16 kHz PCM WAV."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
from config import FFMPEG_DENOISE_FILTER, FFMPEG_FILTERS, HF_CACHE_DIR, OUTPUT_DIR

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)

def preprocess_audio(input_path, denoise=False):
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg was not found. Install it with `winget install --id "
            "Gyan.FFmpeg.Shared -e`, then close and reopen PowerShell and "
            "restart Streamlit. If it is already installed, set "
            "SHRUTILIPI_FFMPEG to the full path of ffmpeg.exe."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    denoise_label = "denoise" if denoise else "nodenoise"
    output_path = OUTPUT_DIR / f"{input_path.stem}_clean_{denoise_label}.wav"
    filter_chain = list(FFMPEG_FILTERS)
    if denoise:
        filter_chain.insert(1, FFMPEG_DENOISE_FILTER)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        ",".join(filter_chain),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    print(f"Cleaning audio: {input_path.name}", flush=True)
    subprocess.run(command, check=True)
    print(f"Clean audio saved: {output_path}", flush=True)
    return output_path


def find_ffmpeg():
    """Find FFmpeg on PATH or in common per-user WinGet install locations."""
    configured = os.environ.get("SHRUTILIPI_FFMPEG")
    candidates = [configured] if configured else []
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(on_path)

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        candidates.extend(
            str(path)
            for package in winget_packages.glob("Gyan.FFmpeg.Shared*")
            for path in package.glob("**/bin/ffmpeg.exe")
        )
        candidates.append(str(Path(local_app_data) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"))

    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(str(Path(program_files) / "WinGet" / "Links" / "ffmpeg.exe"))

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Source audio file; the original is never modified.")
    parser.add_argument("--denoise", action="store_true", help="Enable FFmpeg afftdn noise reduction.")
    args = parser.parse_args()
    try:
        preprocess_audio(args.audio, denoise=args.denoise)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
