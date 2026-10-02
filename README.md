# Shrutilipi

```powershell
cd D:\Shrutilipi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
New-Item -ItemType Directory -Force -Path .pip-cache,.tmp | Out-Null
$env:PIP_CACHE_DIR = "D:\Shrutilipi\.pip-cache"
$env:TEMP = "D:\Shrutilipi\.tmp"
$env:TMP = $env:TEMP
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
winget install --id Gyan.FFmpeg.Shared -e
setx OLLAMA_MODELS "D:\Shrutilipi\models\ollama"
```

After FFmpeg installs, close and reopen PowerShell and restart Streamlit. To
check that it is available, run `ffmpeg -version`. If FFmpeg is installed but
not on PATH, set `SHRUTILIPI_FFMPEG` to its full `ffmpeg.exe` path in the same
PowerShell session before launching the app. Shrutilipi also checks the usual
per-user WinGet install location automatically.

Close and restart the Ollama app after `setx`, then open a new PowerShell window:

```powershell
cd D:\Shrutilipi
.\.venv\Scripts\Activate.ps1
$env:OLLAMA_MODELS = "D:\Shrutilipi\models\ollama"
python src\setup_models.py
python src\check_gpu.py
```

To download the optional transformers fallback model:

```powershell
python src\setup_models.py --with-transformers
```

CLI sample:

```powershell
python src\run_all.py input\class1.mp3 --start 0 --end 5 --backend faster-whisper --notes-lang mixed
```

Start the app with the virtual environment active:

```powershell
streamlit run app.py
```

Open http://localhost:8501.
