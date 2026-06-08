<div align="center">

# VoiceButton

**Push-to-talk voice dictation for Windows**

Hold F9 → speak → release → text appears at your cursor.

Powered by faster-whisper (CUDA), runs as a system tray app.

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows10%2F11-success.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Whisper](https://img.shields.io/badge/engine-faster--whisper-orange.svg)](https://github.com/SYSTRAN/faster-whisper)
[![CUDA](https://img.shields.io/badge/GPU-CUDA-yellow.svg)](#)

</div>

---

## What it does

VoiceButton sits in your system tray and lets you dictate text into **any** application — Telegram, browser, editor, email, anywhere your cursor is.

- **Hold F9** → records from your mic
- **Release F9** → transcribes via Whisper and pastes the text at cursor position
- **Double-click F9** → continuous mode (for long dictation), press F9 again to stop
- **Clipboard preserved** — your existing clipboard content is saved and restored after paste

## Features

- **CUDA-accelerated** transcription via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — ~1 second for 5 seconds of speech on medium model
- **VAD filter** — silence is automatically trimmed
- **Russian punctuation** — Whisper adds commas, periods, and question marks automatically
- **System tray app** — no console window, green/red mic icon shows status
- **Clipboard-safe** — saves and restores your clipboard around paste
- **Configurable** — mic device, model size, language, hotkey

## Download

### Option 1: Ready-to-use EXE (recommended)

Two builds are available depending on your GPU:

| Build | Model | VRAM required | Accuracy | Speed | Best for |
|-------|-------|---------------|----------|-------|----------|
| **voicebutton-medium.exe** | Whisper medium | **6 GB+** | Good | Fast (~1s / 5s speech) | RTX 2060, 3060, 4060 and similar |
| **voicebutton-large.exe** | Whisper large-v3 | **10 GB+** | Best | Slower (~3s / 5s speech) | RTX 3080, 4070, 4080 and similar |

1. Go to [Releases](../../releases)
2. Download the EXE matching your GPU
3. **Right-click → Run as Administrator** (required for keyboard hook)
4. Hold F9 and speak

> **Which one to choose?** If unsure, start with `voicebutton-medium.exe`. It runs on most modern GPUs and accuracy is already very good for Russian and English.

### Option 2: From source

```bash
git clone https://github.com/NIKITANDRRR/voicebuttonwindows.git
cd voicebuttonwindows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python voicebutton.py
```

Run as **Administrator** (required for keyboard hook).

## Requirements

- Windows 10/11
- NVIDIA GPU with CUDA (or set `DEVICE = "cpu"` in config)
- Python 3.12 (if running from source)
- Microphone

## Configuration

Edit the config section at the top of `voicebutton.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_SIZE` | `"medium"` | Whisper model: tiny/base/small/medium/large-v2/large-v3 |
| `DEVICE` | `"cuda"` | `"cuda"` (GPU) or `"cpu"` |
| `COMPUTE_TYPE` | `"float16"` | `float16` for GPU, `int8` for CPU |
| `MIC_DEVICE` | `"Jabra"` | Mic name substring or `None` for default |
| `LANGUAGE` | `None` | `None` = auto-detect, `"ru"` = force Russian |
| `HOTKEY` | `"f9"` | Push-to-talk trigger key |
| `DOUBLE_PRESS_THRESHOLD` | `0.35` | Seconds — max gap for double-click continuous mode |

## Build from source

```bash
pip install pyinstaller
pyinstaller voicebutton.spec
```

Output: `dist/voicebutton.exe`

### Building both releases (medium + large)

```bash
# Build medium (default)
python build_releases.py
```

This script builds both `voicebutton-medium.exe` (Whisper medium) and `voicebutton-large.exe` (Whisper large-v3) into `dist/`.

## How it works

```
F9 held → sounddevice captures 16kHz mono audio
F9 released → faster-whisper transcribes (VAD + Russian punctuation hint)
            → text copied to clipboard
            → Ctrl+V pasted at cursor via keyboard lib
            → original clipboard restored after 300ms
```

## Where are models stored?

On first launch, VoiceButton downloads the Whisper model (~1.5 GB for medium, ~3 GB for large). Models are cached for subsequent runs.

**Default cache location (Windows):**
```
C:\Users\<your-user>\.cache\huggingface\hub\
```

You can change this by setting `MODEL_CACHE_DIR` at the top of `voicebutton.py`.

To free disk space, simply delete the `faster-whisper-*` folder — the model will re-download on next run.

## Logs

When running as tray app, logs are written to:
```
%TEMP%\voicebutton.log
```

Logs auto-rotate: max 1 MB per file, 3 backup files kept (4 MB total).

## License

MIT
