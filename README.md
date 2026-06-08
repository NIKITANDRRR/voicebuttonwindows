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

1. Go to [Releases](../../releases)
2. Download `voicebutton.exe`
3. **Right-click → Run as Administrator** (required for keyboard hook)
4. Hold F9 and speak

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

## How it works

```
F9 held → sounddevice captures 16kHz mono audio
F9 released → faster-whisper transcribes (VAD + Russian punctuation hint)
            → text copied to clipboard
            → Ctrl+V pasted at cursor via keyboard lib
            → original clipboard restored after 300ms
```

## Logs

When running as tray app, logs are written to:
```
%TEMP%\voicebutton.log
```

## License

MIT
