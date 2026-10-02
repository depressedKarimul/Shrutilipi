"""Download local Whisper models and pull the configured Ollama model."""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
from config import (
    FASTER_WHISPER_MEDIUM_REPO,
    FASTER_WHISPER_REPO,
    HF_CACHE_DIR,
    MEDIUM_MODEL_DIR,
    OLLAMA_MODEL_NAME,
    OLLAMA_HOST_SETTING,
    OLLAMA_MODELS_DIR,
    TRANSFORMERS_MODEL_DIR,
    TRANSFORMERS_WHISPER_REPO,
    WHISPER_MODEL_DIR,
)
from ollama_utils import ensure_ollama_server_for_pull, model_is_present

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

import requests
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
from huggingface_hub.file_download import get_hf_file_metadata

def download_file_in_ranges(repo_id, filename, local_dir):
    """Download one large weight file in resumable byte ranges."""
    metadata = get_hf_file_metadata(hf_hub_url(repo_id, filename))
    if not metadata.size:
        raise RuntimeError(f"Hugging Face did not report a size for {repo_id}/{filename}")

    target_path = local_dir / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = target_path.with_name(f".{target_path.name}.partial")
    download_cache = local_dir / ".cache" / "huggingface" / "download"
    stale_partials = list(download_cache.glob(f"*.{metadata.etag}*.incomplete"))
    if not partial_path.exists() and stale_partials:
        best_partial = max(stale_partials, key=lambda path: path.stat().st_size)
        if best_partial.stat().st_size <= metadata.size:
            best_partial.replace(partial_path)

    total_size = metadata.size
    range_size = 64 * 1024 * 1024
    while not partial_path.exists() or partial_path.stat().st_size < total_size:
        start = partial_path.stat().st_size if partial_path.exists() else 0
        end = min(start + range_size - 1, total_size - 1)
        download_url = get_hf_file_metadata(hf_hub_url(repo_id, filename)).location
        headers = {"Range": f"bytes={start}-{end}"}
        expected_bytes = end - start + 1
        last_error = None

        for attempt in range(1, 4):
            before_size = partial_path.stat().st_size if partial_path.exists() else 0
            try:
                with requests.get(
                    download_url,
                    headers=headers,
                    stream=True,
                    timeout=(20, 120),
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(
                            f"Expected HTTP 206 for {repo_id}/{filename} range {start}-{end}, "
                            f"received HTTP {response.status_code}"
                        )
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-"):
                        raise RuntimeError(f"Unexpected Content-Range response: {content_range}")
                    with partial_path.open("ab") as partial_file:
                        for data in response.iter_content(chunk_size=1024 * 1024):
                            if data:
                                partial_file.write(data)
                received_bytes = partial_path.stat().st_size - before_size
                if received_bytes != expected_bytes:
                    raise IOError(
                        f"Short range download for {filename}: received {received_bytes} "
                        f"of {expected_bytes} bytes"
                    )
                last_error = None
                break
            except (requests.RequestException, IOError) as exc:
                last_error = exc
                current_size = partial_path.stat().st_size if partial_path.exists() else 0
                if current_size > before_size:
                    # Resume from the current file size on the next loop.
                    break
                if attempt < 3:
                    print(f"Retrying {filename} range after {type(exc).__name__}: {exc}", flush=True)

        current_size = partial_path.stat().st_size if partial_path.exists() else 0
        if last_error is not None and current_size <= start:
            raise RuntimeError(f"Download failed for {repo_id}/{filename}: {last_error}")

        current_size = partial_path.stat().st_size
        print(
            f"{filename}: {current_size / total_size:.1%} "
            f"({current_size:,} / {total_size:,} bytes)",
            flush=True,
        )

    if partial_path.stat().st_size != total_size:
        raise RuntimeError(f"Incomplete download for {filename}: {partial_path.stat().st_size} bytes")

    partial_path.replace(target_path)
    metadata_path = download_cache / f"{filename.replace('/', '--')}.metadata"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        f"{metadata.commit_hash}\n{metadata.etag}\n{time.time()}\n",
        encoding="utf-8",
    )

    for stale_partial in stale_partials:
        if stale_partial.exists():
            stale_partial.unlink()


def download_huggingface_model(repo_id, local_dir, required_file):
    local_dir.mkdir(parents=True, exist_ok=True)
    marker = local_dir / required_file
    if marker.is_file() and marker.stat().st_size > 0:
        print(f"Already present, skipping {repo_id}: {local_dir}")
        return

    print(f"Downloading {repo_id} into {local_dir}...")

    repo_info = HfApi().model_info(repo_id, files_metadata=True)
    large_weights = [
        item.rfilename
        for item in (repo_info.siblings or [])
        if item.rfilename.lower().endswith((".bin", ".safetensors"))
        and item.size
        and item.size >= 128 * 1024 * 1024
    ]
    if large_weights:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            max_workers=1,
            ignore_patterns=large_weights,
        )
        for filename in large_weights:
            download_file_in_ranges(repo_id, filename, local_dir)

    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), max_workers=1)
    if not marker.is_file() or marker.stat().st_size == 0:
        raise RuntimeError(f"Download completed without expected file: {marker}")
    print(f"Downloaded {repo_id} to {local_dir}")


def pull_ollama_model():
    ollama = shutil.which("ollama")
    if not ollama:
        raise RuntimeError("Ollama CLI was not found. Install Ollama and start the Ollama app first.")

    models = ensure_ollama_server_for_pull(OLLAMA_MODEL_NAME)
    if model_is_present(OLLAMA_MODEL_NAME, models):
        print(f"Ollama model already present, skipping: {OLLAMA_MODEL_NAME}")
        return

    print(f"Pulling Ollama model {OLLAMA_MODEL_NAME}...")
    environment = os.environ.copy()
    environment["OLLAMA_MODELS"] = str(OLLAMA_MODELS_DIR)
    environment["OLLAMA_HOST"] = OLLAMA_HOST_SETTING
    result = subprocess.run([ollama, "pull", OLLAMA_MODEL_NAME], env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Ollama model pull failed. Confirm Ollama is running and OLLAMA_MODELS is set "
            f"to {OLLAMA_MODELS_DIR}, then restart Ollama and retry."
        )
    print(f"Pulled Ollama model: {OLLAMA_MODEL_NAME}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-transformers",
        action="store_true",
        help="Also download openai/whisper-large-v3 for the transformers fallback backend.",
    )
    args = parser.parse_args()

    try:
        download_huggingface_model(FASTER_WHISPER_REPO, WHISPER_MODEL_DIR, "model.bin")
        download_huggingface_model(FASTER_WHISPER_MEDIUM_REPO, MEDIUM_MODEL_DIR, "model.bin")
        if args.with_transformers:
            download_huggingface_model(
                TRANSFORMERS_WHISPER_REPO,
                TRANSFORMERS_MODEL_DIR,
                "model.safetensors",
            )
        pull_ollama_model()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
