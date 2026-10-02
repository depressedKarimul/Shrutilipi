"""Transcribe a cleaned recording with a local speech model."""

import argparse
import json
import os
import sys
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
from config import HF_CACHE_DIR, MEDIUM_MODEL_DIR, OUTPUT_DIR, PROMPT_PATH, TRANSFORMERS_MODEL_DIR, WHISPER_MODEL_DIR

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)

SAMPLE_RATE = 16000
VAD_PARAMETERS = {"threshold": 0.3}


def recording_name(path):
    name = Path(path).stem
    lowered_name = name.lower()
    for suffix in ("_clean_nodenoise", "_clean_denoise", "_clean"):
        if lowered_name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def timestamp(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def read_prompt():
    if PROMPT_PATH.is_file():
        prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        if prompt:
            return prompt
    return "আজকে আমরা machine learning এর overfitting নিয়ে আলোচনা করব।"


def read_audio_window(audio_path, start_seconds, end_seconds):
    import numpy as np

    with wave.open(str(audio_path), "rb") as audio_file:
        if audio_file.getnchannels() != 1 or audio_file.getframerate() != SAMPLE_RATE:
            raise ValueError("Expected preprocessed mono 16 kHz WAV. Run preprocess.py first.")
        total_seconds = audio_file.getnframes() / SAMPLE_RATE
        if start_seconds < 0 or start_seconds >= total_seconds:
            raise ValueError(f"Start time must be between 0 and {total_seconds / 60:.2f} minutes.")
        actual_end = total_seconds if end_seconds is None else min(end_seconds, total_seconds)
        if actual_end <= start_seconds:
            raise ValueError("End time must be greater than start time and within the audio.")
        audio_file.setpos(int(start_seconds * SAMPLE_RATE))
        frame_count = int((actual_end - start_seconds) * SAMPLE_RATE)
        raw_audio = audio_file.readframes(frame_count)

    audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, total_seconds, actual_end - start_seconds


def is_out_of_memory(error):
    message = str(error).lower()
    return any(
        marker in message
        for marker in ("out of memory", "cuda_error_out_of_memory", "cudnn_status_alloc_failed")
    )


def collect_faster_whisper(model_path, compute_type, audio, audio_offset, window_duration, prompt):
    from faster_whisper import WhisperModel

    model = WhisperModel(str(model_path), device="cuda", compute_type=compute_type)
    try:
        segments, info = model.transcribe(
            audio,
            language=None,
            task="transcribe",
            multilingual=True,
            initial_prompt=prompt,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=VAD_PARAMETERS,
        )
    except TypeError as exc:
        if "multilingual" in str(exc).lower() or "unexpected keyword" in str(exc).lower():
            raise RuntimeError(
                "This faster-whisper version does not support multilingual=True. "
                "Upgrade it with: python -m pip install --upgrade faster-whisper"
            ) from exc
        raise

    rows = []
    last_progress = -1
    for segment in segments:
        start = float(segment.start) + audio_offset
        end = float(segment.end) + audio_offset
        text = segment.text.strip()
        if not text:
            continue
        rows.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "avg_logprob": float(segment.avg_logprob),
            }
        )
        progress = min(window_duration, max(0, float(segment.end)))
        progress_seconds = int(progress)
        if progress_seconds > last_progress:
            print(
                f"[progress] {timestamp(progress_seconds)} / {timestamp(window_duration)} audio processed",
                flush=True,
            )
            last_progress = progress_seconds

    print(f"Detected language: {info.language} ({info.language_probability:.2f})", flush=True)
    return rows


def transcribe_faster_whisper(
    audio,
    audio_offset,
    window_duration,
    prompt,
    allow_cpu=False,
    skip_cuda=False,
):
    candidates = [
        (WHISPER_MODEL_DIR, "int8_float16", "large-v3, int8_float16"),
        (WHISPER_MODEL_DIR, "int8", "large-v3, int8"),
        (MEDIUM_MODEL_DIR, "int8_float16", "medium, int8_float16"),
        (MEDIUM_MODEL_DIR, "int8", "medium, int8"),
    ]
    errors = []
    if not skip_cuda:
        for index, (model_path, compute_type, label) in enumerate(candidates):
            if not model_path.is_dir():
                errors.append(f"Missing local model directory: {model_path}")
                continue
            if index > 0:
                print(f"Retrying with {label} after the previous model attempt failed.", flush=True)
            try:
                return collect_faster_whisper(
                    model_path,
                    compute_type,
                    audio,
                    audio_offset,
                    window_duration,
                    prompt,
                )
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                if not is_out_of_memory(exc):
                    break
                print(f"GPU out of memory with {label}; trying the next smaller setting.", flush=True)

    if allow_cpu:
        print(
            "!!! CPU FALLBACK: all requested CUDA model attempts failed; "
            "transcribing on CPU with faster-whisper int8. This will be much slower.",
            flush=True,
        )
        try:
            from faster_whisper import WhisperModel

            model_path = WHISPER_MODEL_DIR if WHISPER_MODEL_DIR.is_dir() else MEDIUM_MODEL_DIR
            model = WhisperModel(str(model_path), device="cpu", compute_type="int8")
            segments, info = model.transcribe(
                audio,
                language=None,
                task="transcribe",
                multilingual=True,
                initial_prompt=prompt,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=VAD_PARAMETERS,
            )
            rows = []
            for segment in segments:
                rows.append(
                    {
                        "start": float(segment.start) + audio_offset,
                        "end": float(segment.end) + audio_offset,
                        "text": segment.text.strip(),
                        "avg_logprob": float(segment.avg_logprob),
                    }
                )
                print(
                    f"[progress] {timestamp(segment.end)} / {timestamp(window_duration)} audio processed",
                    flush=True,
                )
            print(f"Detected language: {info.language} ({info.language_probability:.2f})", flush=True)
            return rows
        except Exception as exc:
            errors.append(f"CPU faster-whisper: {type(exc).__name__}: {exc}")

    raise RuntimeError("faster-whisper failed:\n" + "\n".join(errors))


def transcribe_transformers(audio, audio_offset, window_duration, prompt):
    import numpy as np
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    if not TRANSFORMERS_MODEL_DIR.is_dir():
        raise RuntimeError(
            "The transformers Whisper model is not downloaded. Run "
            "python src\\setup_models.py --with-transformers first."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA for the transformers backend.")

    processor = AutoProcessor.from_pretrained(str(TRANSFORMERS_MODEL_DIR), local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(TRANSFORMERS_MODEL_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda")
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0,
        torch_dtype=torch.float16,
    )
    prompt_ids = processor.get_prompt_ids(prompt) if prompt else None
    rows = []
    chunk_seconds = 30
    for start_sample in range(0, len(audio), chunk_seconds * SAMPLE_RATE):
        end_sample = min(len(audio), start_sample + chunk_seconds * SAMPLE_RATE)
        chunk = audio[start_sample:end_sample]
        generate_kwargs = {"task": "transcribe", "language": None}
        if prompt_ids:
            generate_kwargs["prompt_ids"] = prompt_ids
        result = asr(
            {"raw": np.asarray(chunk, dtype=np.float32), "sampling_rate": SAMPLE_RATE},
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
        chunk_base = start_sample / SAMPLE_RATE + audio_offset
        returned_chunks = result.get("chunks") or []
        if returned_chunks:
            for item in returned_chunks:
                text = item.get("text", "").strip()
                if not text:
                    continue
                local_start, local_end = item.get("timestamp", (0.0, len(chunk) / SAMPLE_RATE))
                rows.append(
                    {
                        "start": chunk_base + (local_start or 0.0),
                        "end": chunk_base + (local_end or len(chunk) / SAMPLE_RATE),
                        "text": text,
                        "avg_logprob": None,
                    }
                )
        elif result.get("text", "").strip():
            rows.append(
                {
                    "start": chunk_base,
                    "end": chunk_base + len(chunk) / SAMPLE_RATE,
                    "text": result["text"].strip(),
                    "avg_logprob": None,
                }
            )
        progress = end_sample / SAMPLE_RATE
        print(f"[progress] {timestamp(progress)} / {timestamp(window_duration)} audio processed", flush=True)
    return rows


def collapse_repetitions(rows):
    output = []
    index = 0
    while index < len(rows):
        key = " ".join(rows[index]["text"].lower().split())
        end = index + 1
        while end < len(rows) and " ".join(rows[end]["text"].lower().split()) == key:
            end += 1
        count = end - index
        if count >= 3:
            kept = dict(rows[index])
            kept["repetition_filtered"] = True
            kept["repetition_count"] = count
            output.append(kept)
            print(
                f"Repetition filter: collapsed {count} identical consecutive lines "
                f"at {timestamp(kept['start'])}.",
                flush=True,
            )
        else:
            for row in rows[index:end]:
                kept = dict(row)
                kept["repetition_filtered"] = False
                kept["repetition_count"] = 1
                output.append(kept)
        index = end
    return output


def write_results(name, rows, backend):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path = OUTPUT_DIR / f"{name}_transcript.txt"
    json_path = OUTPUT_DIR / f"{name}_transcript.json"
    text_lines = []
    for row in rows:
        line = f"[{timestamp(row['start'])}] {row['text']}"
        if row.get("repetition_filtered"):
            line += f" [repetition filtered: {row['repetition_count']} identical lines]"
        text_lines.append(line)
    transcript_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {"backend": backend, "segments": rows},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Transcript saved: {transcript_path}", flush=True)
    print(f"Transcript JSON saved: {json_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Cleaned 16 kHz mono WAV file.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in minutes.")
    parser.add_argument("--end", type=float, help="End time in minutes; defaults to the end of the audio.")
    parser.add_argument(
        "--backend",
        choices=("faster-whisper", "transformers", "auto"),
        default="faster-whisper",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.is_file():
        print(f"ERROR: audio file not found: {audio_path}", file=sys.stderr)
        return 1
    try:
        start_seconds = args.start * 60
        end_seconds = None if args.end is None else args.end * 60
        audio, total_seconds, window_duration = read_audio_window(audio_path, start_seconds, end_seconds)
        print(
            f"Audio window: {timestamp(start_seconds)} to "
            f"{timestamp(start_seconds + window_duration)} of {timestamp(total_seconds)}",
            flush=True,
        )
        prompt = read_prompt()
        backend = args.backend
        rows = None

        if backend in ("faster-whisper", "auto"):
            try:
                print("Using faster-whisper on CUDA...", flush=True)
                rows = transcribe_faster_whisper(audio, start_seconds, window_duration, prompt)
                backend = "faster-whisper"
            except Exception as exc:
                if args.backend == "faster-whisper":
                    print(f"faster-whisper CUDA failed: {exc}", flush=True)
                    rows = transcribe_faster_whisper(
                        audio,
                        start_seconds,
                        window_duration,
                        prompt,
                        allow_cpu=True,
                        skip_cuda=True,
                    )
                    backend = "faster-whisper-cpu"
                else:
                    print(f"faster-whisper CUDA check failed: {exc}", flush=True)
                    print("Trying the transformers CUDA backend next.", flush=True)

        if rows is None and args.backend in ("transformers", "auto"):
            try:
                print("Using transformers Whisper on PyTorch CUDA (fp16)...", flush=True)
                rows = transcribe_transformers(audio, start_seconds, window_duration, prompt)
                backend = "transformers"
            except Exception as exc:
                if args.backend == "transformers":
                    print(f"Transformers CUDA failed: {exc}", flush=True)
                else:
                    print(f"Transformers CUDA check failed: {exc}", flush=True)

        if rows is None:
            print(
                "!!! CPU FALLBACK: faster-whisper and transformers CUDA paths failed; "
                "transcribing on CPU with faster-whisper int8. This will be much slower.",
                flush=True,
            )
            rows = transcribe_faster_whisper(
                audio,
                start_seconds,
                window_duration,
                prompt,
                allow_cpu=True,
                skip_cuda=True,
            )
            backend = "faster-whisper-cpu"

        write_results(recording_name(audio_path), collapse_repetitions(rows), backend)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
