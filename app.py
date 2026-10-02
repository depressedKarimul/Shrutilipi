"""Small Streamlit frontend for the Shrutilipi command-line pipeline."""

import hashlib
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT / "src"))

from config import (
    HF_CACHE_DIR,
    INPUT_DIR,
    OLLAMA_MODEL_NAME,
    OLLAMA_MODELS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
)
from ollama_utils import ensure_ollama, get_server_models, model_is_present

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)

import streamlit as st


ALLOWED_AUDIO_TYPES = ("mp3", "wav", "m4a", "aac", "ogg", "flac", "mp4")
PLACEHOLDER = "Select a previous recording"


def save_upload(uploaded_file):
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_name = Path(uploaded_file.name).name
    safe_name = Path(original_name)
    stem = safe_name.stem or "recording"
    suffix = safe_name.suffix
    digest = hashlib.sha256()
    uploaded_file.seek(0)
    while True:
        block = uploaded_file.read(8 * 1024 * 1024)
        if not block:
            break
        digest.update(block)
    uploaded_file.seek(0)
    token = f"{original_name}:{uploaded_file.size}:{digest.hexdigest()}"

    if st.session_state.get("last_upload_token") == token:
        existing = st.session_state.get("last_upload_path")
        if existing and Path(existing).is_file():
            return Path(existing)

    candidate = INPUT_DIR / f"{stem}{suffix}"
    number = 1
    while candidate.exists():
        candidate = INPUT_DIR / f"{stem}_{number}{suffix}"
        number += 1
    with candidate.open("xb") as output_file:
        shutil.copyfileobj(uploaded_file, output_file, length=8 * 1024 * 1024)
    uploaded_file.seek(0)

    st.session_state["last_upload_token"] = token
    st.session_state["last_upload_path"] = str(candidate)
    st.session_state["active_recording"] = candidate.stem
    return candidate


def recordings_with_outputs():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for audio_path in INPUT_DIR.iterdir():
        if not audio_path.is_file():
            continue
        stem = audio_path.stem
        if (OUTPUT_DIR / f"{stem}_notes.md").is_file() or (OUTPUT_DIR / f"{stem}_transcript.txt").is_file():
            items.append(audio_path)
    return sorted(items, key=lambda path: path.stat().st_mtime, reverse=True)


def gpu_status():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return "GPU: " + " | ".join(lines) if lines else "GPU info not available"
    except Exception:
        return "GPU info not available"


def ollama_status():
    """Inspect the local Ollama API without starting or stopping its server."""
    models = get_server_models()
    if models is None:
        return "Ollama: not running"
    if model_is_present(OLLAMA_MODEL_NAME, models):
        return f"Ollama: running; {OLLAMA_MODEL_NAME} available"
    return f"Ollama: running; {OLLAMA_MODEL_NAME} missing"


def run_pipeline(audio_path, short_test, start_minute, end_minute, backend, notes_lang, force, denoise):
    command = [sys.executable, "src/run_all.py", str(audio_path), "--backend", backend, "--notes-lang", notes_lang]
    if short_test:
        command.extend(["--start", str(start_minute), "--end", str(end_minute)])
    if force:
        command.append("--force")
    if denoise:
        command.append("--denoise")

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["HF_HOME"] = str(HF_CACHE_DIR)
    environment["HF_HUB_CACHE"] = str(HF_CACHE_DIR)

    last_lines = deque(maxlen=30)
    status = st.status("Running Shrutilipi…", expanded=True)
    log_box = st.empty()
    st.session_state["run_active"] = True
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        st.session_state["pipeline_process_id"] = process.pid
        if process.stdout is not None:
            for line in process.stdout:
                last_lines.append(line.rstrip())
                log_box.code("\n".join(last_lines), language=None)
        return_code = process.wait()
    except Exception as exc:
        status.update(label="Pipeline failed", state="error", expanded=True)
        st.error(f"Could not start the pipeline: {exc}")
        return False
    finally:
        st.session_state["run_active"] = False
        st.session_state.pop("pipeline_process_id", None)

    if return_code == 0:
        status.update(label="Pipeline complete", state="complete", expanded=False)
        st.success("Transcription and notes are ready.")
        st.session_state["active_recording"] = Path(audio_path).stem
        return True

    status.update(label=f"Pipeline failed (exit code {return_code})", state="error", expanded=True)
    st.error("The pipeline failed. Last output lines:")
    st.code("\n".join(last_lines) or "No output was produced.", language=None)
    return False


def read_needs_checking(notes_text):
    match = re.search(r"(?im)^#{1,6}\s+Needs checking\s*$", notes_text)
    if not match:
        return None
    remainder = notes_text[match.end():]
    following_heading = re.search(r"(?m)^#{1,6}\s+", remainder)
    section = remainder[:following_heading.start()] if following_heading else remainder
    return [
        line.strip()
        for line in section.splitlines()
        if "[unclear]" in line.lower() and re.search(r"\[\d{2,}:\d{2}\]", line)
    ]


def show_results(recording_stem):
    notes_path = OUTPUT_DIR / f"{recording_stem}_notes.md"
    transcript_path = OUTPUT_DIR / f"{recording_stem}_transcript.txt"
    if not notes_path.is_file() and not transcript_path.is_file():
        st.info("No results are available for this recording yet.")
        return

    notes_text = notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""
    transcript_text = transcript_path.read_text(encoding="utf-8") if transcript_path.is_file() else ""
    notes_tab, transcript_tab, checking_tab = st.tabs(["Notes", "Transcript", "Needs checking"])

    with notes_tab:
        if notes_text:
            st.markdown(notes_text)
            st.download_button(
                "Download notes",
                data=notes_text.encode("utf-8"),
                file_name=notes_path.name,
                mime="text/markdown; charset=utf-8",
            )
        else:
            st.info("Notes are not available yet.")

    with transcript_tab:
        if transcript_text:
            st.text_area("Transcript", value=transcript_text, height=520, disabled=True, label_visibility="collapsed")
            st.download_button(
                "Download transcript",
                data=transcript_text.encode("utf-8"),
                file_name=transcript_path.name,
                mime="text/plain; charset=utf-8",
            )
        else:
            st.info("Transcript is not available yet.")

    with checking_tab:
        unclear_lines = read_needs_checking(notes_text) if notes_text else None
        if unclear_lines is None:
            st.info("There is no Needs checking section in the notes.")
        elif unclear_lines:
            st.markdown("\n".join(f"- {line.lstrip('-* ')}" for line in unclear_lines))
        else:
            st.info("No timestamped [unclear] lines were listed.")


def main():
    st.set_page_config(page_title="Shrutilipi", page_icon="📝", layout="wide")
    st.title("Shrutilipi")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    st.session_state.setdefault("run_active", False)

    with st.sidebar:
        st.header("Previous recordings")
        previous_items = recordings_with_outputs()
        labels = [PLACEHOLDER] + [path.name for path in previous_items]
        previous_index = 0
        active = st.session_state.get("active_recording")
        for index, path in enumerate(previous_items, start=1):
            if path.stem == active:
                previous_index = index
                break
        selected_previous = st.selectbox(
            "Previous recording",
            labels,
            index=previous_index,
            label_visibility="collapsed",
        )
        st.caption(gpu_status())
        st.caption(ollama_status())
        if st.button("Start Ollama (D: models)"):
            try:
                ensure_ollama(OLLAMA_MODEL_NAME)
                st.success(f"Ollama is ready; models folder: {OLLAMA_MODELS_DIR}")
            except Exception as exc:
                st.error(str(exc))

    uploaded = st.file_uploader("Upload a classroom recording", type=ALLOWED_AUDIO_TYPES)
    audio_path = None
    if uploaded is not None:
        try:
            audio_path = save_upload(uploaded)
            st.caption(f"Saved as {audio_path.relative_to(PROJECT_ROOT)}")
        except Exception as exc:
            st.error(f"Could not save the upload: {exc}")
    else:
        st.session_state["last_upload_token"] = None
        st.session_state.pop("last_upload_path", None)

    with st.expander("Options", expanded=False):
        short_test = st.checkbox("Test on a short part only")
        if short_test:
            start_minute = st.number_input("Start minute", min_value=0.0, value=0.0, step=1.0)
            end_minute = st.number_input("End minute", min_value=0.1, value=5.0, step=1.0)
        else:
            start_minute = 0.0
            end_minute = 5.0
        backend = st.selectbox(
            "Backend",
            ["faster-whisper", "auto", "transformers"],
            index=0,
        )
        force = st.checkbox("Re-run everything")
        denoise = st.checkbox("Denoise audio (afftdn)")
        notes_label = st.selectbox(
            "Notes language",
            [
                "English headings + Bangla where lecturer used Bangla",
                "English only",
            ],
            index=0,
        )

    notes_lang = "mixed" if notes_label.startswith("English headings") else "english"
    run_disabled = st.session_state["run_active"] or audio_path is None
    if st.button("Transcribe and make notes", type="primary", disabled=run_disabled):
        if end_minute <= start_minute and short_test:
            st.error("End minute must be greater than start minute.")
        else:
            run_pipeline(audio_path, short_test, start_minute, end_minute, backend, notes_lang, force, denoise)

    selected_stem = None
    if selected_previous != PLACEHOLDER:
        selected_stem = Path(selected_previous).stem
    elif st.session_state.get("active_recording"):
        selected_stem = st.session_state["active_recording"]
    if selected_stem:
        st.subheader(selected_stem.replace("_", " "))
        show_results(selected_stem)


if __name__ == "__main__":
    main()
