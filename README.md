# VoiceButton

Push-to-talk voice transcription for Windows.

Hold **F12**, speak, release — text appears in the active window via Ctrl+V.

## Features
- CUDA-accelerated transcription via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- VAD (Voice Activity Detection) filter — ignores silence
- Clipboard paste — instant text input into any app
- Configurable mic device (substring match), model size, language

## Requirements
- Windows 10/11
- NVIDIA GPU + CUDA (or set `DEVICE = "cpu"`)
- Python 3.12
- Microphone

## Install & Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install faster-whisper keyboard sounddevice numpy pyautogui pyperclip
python voicebutton.py
```

Run as **Administrator** (required for keyboard hook).

## Config (top of voicebutton.py)
| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_SIZE` | `"medium"` | Whisper model: tiny/base/small/medium/large-v2/large-v3 |
| `DEVICE` | `"cuda"` | `"cuda"` or `"cpu"` |
| `COMPUTE_TYPE` | `"float16"` | `float16` for GPU, `int8` for CPU |
| `MIC_DEVICE` | `"Jabra"` | Mic name substring or `None` for default |
| `LANGUAGE` | `None` | `None` = auto-detect, `"ru"` = force Russian |
| `HOTKEY` | `"f12"` | Push-to-talk key |

## Build EXE
```bash
pip install pyinstaller
pyinstaller voicebutton.spec
```
Output: `dist/voicebutton.exe`
