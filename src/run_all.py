"""Run preprocessing, transcription, and note generation for one recording."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
from config import HF_CACHE_DIR, OUTPUT_DIR

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)

def run_step(label, command, environment):
    print(f"\n=== {label} ===", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}.")


def normalize_transcript_names(clean_audio, transcript, transcript_json):
    """Adopt transcript filenames produced by older clean-name handling."""
    legacy_paths = (
        (OUTPUT_DIR / f"{clean_audio.stem}_transcript.txt", transcript),
        (OUTPUT_DIR / f"{clean_audio.stem}_transcript.json", transcript_json),
    )
    for legacy_path, expected_path in legacy_paths:
        if legacy_path != expected_path and legacy_path.is_file() and not expected_path.exists():
            legacy_path.replace(expected_path)
            print(f"Adjusted transcript filename: {expected_path.name}", flush=True)


def run_pipeline(audio_path, start, end, backend, notes_lang, force=False, denoise=False):
    audio_path = Path(audio_path)
    if not audio_path.is_absolute():
        audio_path = (PROJECT_ROOT / audio_path).resolve()
    else:
        audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = audio_path.stem
    denoise_label = "denoise" if denoise else "nodenoise"
    clean_audio = OUTPUT_DIR / f"{name}_clean_{denoise_label}.wav"
    transcript = OUTPUT_DIR / f"{name}_transcript.txt"
    transcript_json = OUTPUT_DIR / f"{name}_transcript.json"
    notes = OUTPUT_DIR / f"{name}_notes.md"
    normalize_transcript_names(clean_audio, transcript, transcript_json)

    environment = os.environ.copy()
    environment["HF_HOME"] = str(HF_CACHE_DIR)
    environment["HF_HUB_CACHE"] = str(HF_CACHE_DIR)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"

    if force or not clean_audio.is_file():
        preprocess_command = [sys.executable, "src/preprocess.py", str(audio_path)]
        if denoise:
            preprocess_command.append("--denoise")
        run_step(
            "Preprocess",
            preprocess_command,
            environment,
        )
    else:
        print(f"Skipping preprocessing; found {clean_audio}", flush=True)

    if force or not (transcript.is_file() and transcript_json.is_file()):
        command = [
            sys.executable,
            "src/transcribe.py",
            str(clean_audio),
            "--start",
            str(start),
            "--backend",
            backend,
        ]
        if end is not None:
            command.extend(["--end", str(end)])
        run_step("Transcribe", command, environment)
        normalize_transcript_names(clean_audio, transcript, transcript_json)
    else:
        print(f"Skipping transcription; found {transcript}", flush=True)

    if force or not notes.is_file():
        run_step(
            "Make notes",
            [
                sys.executable,
                "src/make_notes.py",
                str(transcript),
                "--notes-lang",
                notes_lang,
            ],
            environment,
        )
    else:
        print(f"Skipping note generation; found {notes}", flush=True)

    print("\nPipeline complete.", flush=True)
    print(f"Clean audio: {clean_audio}", flush=True)
    print(f"Transcript: {transcript}", flush=True)
    print(f"Transcript JSON: {transcript_json}", flush=True)
    print(f"Notes: {notes}", flush=True)
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Audio file to process.")
    parser.add_argument("--start", type=float, default=0.0, help="Start minute for a short sample.")
    parser.add_argument("--end", type=float, help="End minute for a short sample.")
    parser.add_argument(
        "--backend",
        choices=("auto", "faster-whisper", "transformers"),
        default="faster-whisper",
    )
    parser.add_argument("--notes-lang", choices=("mixed", "english"), default="mixed")
    parser.add_argument("--force", action="store_true", help="Re-run all stages even if outputs exist.")
    parser.add_argument("--denoise", action="store_true", help="Enable FFmpeg afftdn noise reduction.")
    args = parser.parse_args()

    try:
        run_pipeline(
            args.audio,
            args.start,
            args.end,
            args.backend,
            args.notes_lang,
            args.force,
            args.denoise,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
