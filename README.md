# Shrutilipi

Shrutilipi turns classroom recordings into timestamped transcripts and structured study notes. Audio processing, transcription, and note generation run locally using FFmpeg, faster-whisper, and Ollama.

## Features

- Streamlit interface for uploading and processing recordings
- Timestamped transcription with the faster-whisper CUDA backend by default
- Optional audio denoising
- Structured notes generated locally with Ollama and `qwen2.5:7b`
- English headings with the lecturer’s Bangla explanations preserved
- Reuse of existing stage outputs; `--force` reruns the pipeline
- Project models, caches, inputs, and outputs stored under `D:\Shrutilipi`

## Requirements

- Windows 11
- Python 3.13 recommended
- NVIDIA GPU and a current NVIDIA driver for CUDA transcription
- FFmpeg
- Ollama for local note generation
- Internet access for the initial package and model downloads

## Installation

Open PowerShell and create the virtual environment:

```powershell
cd D:\Shrutilipi
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Keep package downloads and temporary files on the project drive, then install the Python dependencies and CUDA-enabled PyTorch:

```powershell
New-Item -ItemType Directory -Force -Path .pip-cache,.tmp | Out-Null
$env:PIP_CACHE_DIR = "D:\Shrutilipi\.pip-cache"
$env:TEMP = "D:\Shrutilipi\.tmp"
$env:TMP = $env:TEMP

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Install FFmpeg if it is not already available:

```powershell
winget install --id Gyan.FFmpeg.Shared -e
```

Close and reopen PowerShell after installation. Verify FFmpeg with `ffmpeg -version`. If FFmpeg is installed but cannot be found, set `SHRUTILIPI_FFMPEG` to the full path to `ffmpeg.exe` before launching Shrutilipi. The app also checks the usual per-user WinGet location.

Install Ollama and confirm its `ollama` command is available in PowerShell. Shrutilipi manages the local server and configures it to use `D:\Shrutilipi\models\ollama`; a global `OLLAMA_MODELS` setting is not required.

## Download models

With the virtual environment active, run:

```powershell
cd D:\Shrutilipi
python src\setup_models.py
```

This downloads the faster-whisper large-v3 and medium models, and pulls `qwen2.5:7b` into the project’s Ollama model directory. Existing models are skipped. The transformers Whisper model is optional and is not downloaded by default:

```powershell
python src\setup_models.py --with-transformers
```

To restart Ollama using the project model directory and check which models the server sees:

```powershell
python src\ollama_utils.py --restart
$env:OLLAMA_MODELS = "D:\Shrutilipi\models\ollama"
$env:OLLAMA_HOST = "127.0.0.1:11434"
ollama list
```

The helper starts Ollama in the background and writes server output to `output\ollama_server.log`.

## Check CUDA

```powershell
python src\check_gpu.py
```

The check reports PyTorch CUDA availability and runs a short faster-whisper GPU inference check. The tiny check model may be downloaded if it is not already cached.

## Use the web app

Activate the virtual environment and start Streamlit:

```powershell
cd D:\Shrutilipi
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload a recording, choose any options, and select **Transcribe and make notes**. The sidebar displays GPU and Ollama status. **Start Ollama (D: models)** starts or reuses the local server.

## Use the command line

Place a recording in `input` or provide its full path. For example, transcribe a five-minute sample:

```powershell
python src\run_all.py input\classroom.mp3 --start 0 --end 5
```

Run the entire recording:

```powershell
python src\run_all.py input\classroom.mp3
```

Enable FFmpeg denoising with `--denoise`. Denoising is off by default, and both cleaned audio variants can be kept side by side:

```powershell
python src\run_all.py input\classroom.mp3 --denoise
```

The default transcription backend is `faster-whisper`. Other available backends are `auto` and `transformers`; the transformers backend requires the optional model download shown above.

Use `--force` to rerun preprocessing, transcription, and note generation even when outputs already exist:

```powershell
python src\run_all.py input\classroom.mp3 --force
```

To process an existing transcript into notes without running the audio stages:

```powershell
python src\make_notes.py "output\classroom_transcript.txt"
```

## Outputs

All generated files are saved in `output`:

- `*_clean_nodenoise.wav` or `*_clean_denoise.wav` — preprocessed audio
- `*_transcript.txt` — timestamped transcript
- `*_transcript.json` — transcript segments and metadata
- `*_notes.md` — structured class notes
- `*_chunks\` — intermediate notes for each transcript chunk

The pipeline skips preprocessing, transcription, or note generation when that stage’s output already exists. It resumes at the first missing stage. Use `--force` to rebuild existing results.

## Project layout

```text
D:\Shrutilipi\
├── app.py                 # Streamlit interface
├── input\                 # Source recordings
├── models\                # Whisper models, Ollama data, and caches
├── output\                # Clean audio, transcripts, logs, and notes
├── src\
│   ├── config.py           # Shared paths and settings
│   ├── preprocess.py       # FFmpeg audio preparation
│   ├── transcribe.py       # Local speech recognition
│   ├── make_notes.py       # Ollama-powered notes
│   ├── ollama_utils.py     # Local Ollama server management
│   ├── run_all.py          # End-to-end pipeline
│   └── setup_models.py     # Model downloads
└── requirements.txt
```

## Troubleshooting

**FFmpeg was not found**  
Install FFmpeg, reopen PowerShell, and check `ffmpeg -version`. If needed, set `SHRUTILIPI_FFMPEG` to the full executable path.

**Ollama does not show `qwen2.5:7b`**  
Run `python src\ollama_utils.py --restart`, then check the server with `ollama list`. If the model is absent, run `python src\setup_models.py` to pull it into the project’s D: model directory.

**CUDA is unavailable**  
Activate the project virtual environment and run `python src\check_gpu.py`. Confirm the NVIDIA driver is installed and that the CUDA-enabled PyTorch package is installed in `.venv`.

**The app shows no results after a failure**  
Review the pipeline output in the app. After resolving the reported issue, run the same recording again; completed stages with existing outputs will be skipped.
