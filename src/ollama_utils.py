"""Manage Shrutilipi's Ollama server and inspect its local model list."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from config import (
    OLLAMA_HOST_SETTING,
    OLLAMA_MODEL_NAME,
    OLLAMA_MODELS_DIR,
    OLLAMA_POLL_INTERVAL_SECONDS,
    OLLAMA_REQUEST_TIMEOUT_SECONDS,
    OLLAMA_RESTART_WAIT_SECONDS,
    OLLAMA_SERVER_LOG,
    OLLAMA_SERVER_URL,
    OLLAMA_STARTUP_TIMEOUT_SECONDS,
)


def get_server_models(timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS):
    """Return the server's model names, or None when its API is unavailable."""
    try:
        with urlopen(f"{OLLAMA_SERVER_URL}/api/tags", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, ValueError):
        return None

    return sorted(
        name
        for item in data.get("models", [])
        if (name := item.get("name") or item.get("model"))
    )


def model_is_present(model, models):
    """Match Ollama's tagged name, accepting an omitted :latest tag."""
    if models is None:
        return False
    base_name = model if ":" in model else f"{model}:latest"
    return any(name == model or name == base_name for name in models)


def stop_ollama_server():
    """Stop Windows Ollama processes; missing processes are harmless."""
    if os.name != "nt":
        raise RuntimeError("Stopping Ollama by process name is supported on Windows only.")

    for image_name in ("ollama app.exe", "ollama.exe"):
        result = subprocess.run(
            ["taskkill", "/f", "/im", image_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stdout or "") + (result.stderr or "")
            lowered = message.lower()
            if "not found" not in lowered and "could not be found" not in lowered and "not running" not in lowered:
                raise RuntimeError(f"Could not stop {image_name}: {message.strip() or 'taskkill failed'}")
    time.sleep(OLLAMA_RESTART_WAIT_SECONDS)


def start_ollama_server(wait=True, allow_existing=True):
    """Start Ollama with its model store pointed at the project D: folder."""
    models = get_server_models()
    if models is not None:
        if allow_existing:
            return models
        raise RuntimeError(
            f"An Ollama server is still answering at {OLLAMA_SERVER_URL} after it was stopped. "
            "Close the other server process and retry."
        )

    ollama_executable = shutil.which("ollama")
    if not ollama_executable:
        raise RuntimeError("Ollama CLI was not found on PATH. Install Ollama first.")

    OLLAMA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OLLAMA_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OLLAMA_MODELS"] = str(OLLAMA_MODELS_DIR)
    environment["OLLAMA_HOST"] = OLLAMA_HOST_SETTING
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

    with OLLAMA_SERVER_LOG.open("a", encoding="utf-8") as log_file:
        subprocess.Popen(
            [ollama_executable, "serve"],
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creation_flags,
            cwd=str(OLLAMA_MODELS_DIR.parent.parent),
        )

    if not wait:
        return None

    deadline = time.monotonic() + OLLAMA_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        models = get_server_models()
        if models is not None:
            return models
        time.sleep(OLLAMA_POLL_INTERVAL_SECONDS)
    raise RuntimeError(
        f"Ollama did not answer at {OLLAMA_SERVER_URL} within "
        f"{OLLAMA_STARTUP_TIMEOUT_SECONDS} seconds. See {OLLAMA_SERVER_LOG}."
    )


def restart_ollama_server():
    """Replace any running tray or CLI instance with the project's server."""
    if os.name == "nt":
        stop_ollama_server()
    elif get_server_models() is not None:
        raise RuntimeError("Cleanly restarting the Ollama process is supported on Windows only.")
    return start_ollama_server(wait=True, allow_existing=False)


def ensure_ollama(model):
    """Ensure the requested model is served, restarting a foreign server if needed."""
    models = get_server_models()
    if models is not None and model_is_present(model, models):
        print(f"Ollama models folder: {OLLAMA_MODELS_DIR}", flush=True)
        return True

    if models is not None:
        # A responding server without our model likely belongs to the tray app
        # and is reading the user's default models directory.
        stop_ollama_server()

    models = start_ollama_server(wait=True)
    print(f"Ollama models folder: {OLLAMA_MODELS_DIR}", flush=True)
    if not model_is_present(model, models):
        present = ", ".join(models) if models else "none"
        raise RuntimeError(
            f"Ollama is using OLLAMA_MODELS={OLLAMA_MODELS_DIR}, but "
            f"{model!r} is not present (found: {present}). To pull it into D: models, "
            "run: python src\\setup_models.py"
        )
    return True


def ensure_ollama_server_for_pull(model):
    """Start the project server and move away from a server missing this model."""
    models = get_server_models()
    if models is not None and not model_is_present(model, models):
        stop_ollama_server()
        models = None
    if models is None:
        models = start_ollama_server(wait=True)
    print(f"Ollama models folder: {OLLAMA_MODELS_DIR}", flush=True)
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", action="store_true", help="Restart Ollama using D: project models.")
    args = parser.parse_args()
    try:
        models = restart_ollama_server() if args.restart else start_ollama_server(wait=True)
        print(f"Ollama models folder: {OLLAMA_MODELS_DIR}")
        print("Models: " + (", ".join(models) if models else "none"))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
