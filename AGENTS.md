# AGENTS.md — VoiceButton

## Project
Push-to-talk voice transcription for Windows. Hold F12, speak, release → text pasted into active window.

## Architecture
Single-file app: `voicebutton.py`

Flow: keyboard hook → sd.InputStream capture → faster-whisper transcribe → pyperclip+pyautogui paste

## Key Decisions
- **Hotkey: F12** — Right Alt = AltGr on RU layout (not detectable by `keyboard` lib), Scroll Lock = OS-handled
- **Mic: Jabra Speak 710** — Realtek default records silence. MIC_DEVICE config supports substring match
- **Admin required** — `keyboard` hook needs elevated privileges on Windows
- **VAD model** — `silero_vad_v6.onnx` must be included in PyInstaller datas (spec file)

## Build
```
pyinstaller voicebutton.spec
```
Output: `dist/voicebutton.exe`

## Gotchas
- Run as Administrator — otherwise keyboard hook silently fails
- If "Audio is SILENT" — wrong mic device, change MIC_DEVICE in config
- `vad_filter=True` requires silero_vad onnx in assets — bundled via spec `datas`
- PyInstaller one-file mode: all DLLs packed into single exe, first run slower (extraction)

## Config (top of voicebutton.py)
| Setting | Default | Notes |
|---------|---------|-------|
| MODEL_SIZE | medium | tiny/base/small/medium/large-v2/large-v3 |
| DEVICE | cuda | cpu alternative for no-GPU |
| COMPUTE_TYPE | float16 | int8 for CPU |
| SAMPLE_RATE | 16000 | whisper expects 16kHz |
| MIC_DEVICE | "Jabra" | substring match on device name, None=default |
| LANGUAGE | None | None=auto, "ru"=force Russian |
| BEAM_SIZE | 5 | beam search width |
| HOTKEY | f12 | push-to-talk trigger |
