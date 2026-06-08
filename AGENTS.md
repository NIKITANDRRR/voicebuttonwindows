# AGENTS.md — VoiceButton

## Project
Push-to-talk voice transcription for Windows. Hold F9, speak, release → text pasted into active window at cursor position. Double-click F9 for continuous mode. Runs as system tray icon. Clipboard is preserved.

## Architecture
Single-file app: `voicebutton.py`

Flow: keyboard hook → sd.InputStream capture → faster-whisper transcribe (with Russian punctuation prompt) → save clipboard → pyperclip+Ctrl+V via keyboard lib → restore clipboard

## Key Decisions
- **Hotkey: F9** — Right Shift caused Windows Sticky Keys popup; Right Alt = AltGr on RU (not detectable); Scroll Lock = OS-handled; F9 is unused and clean
- **Double-click F9 = continuous mode** — quick press-release-press within 350ms toggles continuous recording until next F9 press. Implemented via release-timer that gets cancelled on second press
- **System tray** — no console window; green/red mic icon shows status (idle/recording). Exit via tray menu. Logs go to `%TEMP%\voicebutton.log`
- **Russian punctuation** — initial_prompt with sample punctuated Russian text makes Whisper add commas/periods in dictation
- **Cursor-position paste via Ctrl+V** — uses keyboard library (not pyautogui) for key simulation. Clipboard saved before paste, restored 300ms after
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
| HOTKEY | f9 | push-to-talk trigger |
| DOUBLE_PRESS_THRESHOLD | 0.35 | seconds — max gap for double-click continuous mode |
| INITIAL_PROMPT | "Привет, как..." | Russian text hint for punctuation |
| LOG_FILE | %TEMP%\voicebutton.log | log file path |
