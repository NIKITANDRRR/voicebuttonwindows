# AGENTS.md — VoiceButton

## Project
Push-to-talk voice transcription for Windows. Hold F9, speak, release → text pasted into active window at cursor position. F10 toggles continuous mode. Runs as system tray icon. Clipboard is preserved.

## Architecture
Single-file app: `voicebutton.py`

Flow: keyboard hook → sd.InputStream capture → faster-whisper transcribe (with Russian punctuation prompt) → save clipboard → pyperclip+Ctrl+V via keyboard lib → restore clipboard

## Docs
- `SETUP_GUIDE.md` (RU) / `SETUP_GUIDE_EN.md` (EN) — step-by-step computer setup for users: GPU check, NVIDIA driver install, choosing medium vs large build, first run, troubleshooting. Linked from the README header. Keep wording neutral (no "beginners/newbies" framing), formal "вы"/"you" address.

## Key Decisions
- **Hotkey: F9** — Right Shift caused Windows Sticky Keys popup; Right Alt = AltGr on RU (not detectable); Scroll Lock = OS-handled; F9 is unused and clean
- **F10 = continuous mode** — dedicated key (was double-click F9 with a 0.35s wait on EVERY dictation; now release transcribes immediately)
- **F8 = paste from history** — last 10 dictations kept in memory; each further F8 press goes one step back, new dictation resets position. Tray "History" submenu lists them too. Covers the "focus moved, text landed nowhere" case
- **Mic level overlay + beeps** — tkinter topmost bar (bottom-right) fed by RMS in the portaudio callback; green = voice, red = clipping. winsound beep 880 Hz on record start, 440 Hz on stop. Both off-able via ini [feedback]
- **Per-machine config: voicebutton.ini** — optional INI next to script/exe overrides code defaults (gitignored; see voicebutton.ini.example). Keeps repo and installed copies from diverging: work laptop = small/cpu, home PC = medium/cuda, same code
- **Domain vocabulary** — VOCABULARY terms (Метраникс, ТФВС, САКВ...) appended to initial_prompt; fixes "Metronics"-type errors. Overridable via ini `terms`
- **Model warmup at startup** — dummy transcribe on 1s of zeros right after load, so first real dictation isn't slow
- **System tray** — no console window; green/red mic icon shows status (idle/recording). Exit via tray menu. Logs go to `%TEMP%\voicebutton.log`
- **Russian forced by default** — LANGUAGE="ru"; "auto" caused English background audio (YouTube) to leak into dictation
- **Russian punctuation** — initial_prompt with sample punctuated Russian text makes Whisper add commas/periods in dictation
- **Cursor-position paste via Ctrl+V** — uses keyboard library (not pyautogui) for key simulation. Clipboard saved before paste, restored 300ms after
- **Mic: Jabra Speak 710** — Realtek default records silence. MIC_DEVICE config supports substring match
- **Admin required** — `keyboard` hook needs elevated privileges on Windows
- **Auto CPU fallback** — if no NVIDIA driver/CUDA, программа сама переключается на CPU (int8). Пользователь видит в трее "CPU mode (no GPU)"
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
- CUDA Toolkit is NOT required for the ready EXE — the NVIDIA driver alone is enough; cuDNN/cuBLAS ship inside PyInstaller bundle (from source they must be installed separately, see faster-whisper GPU docs)

## Config (code defaults, overridden by voicebutton.ini)
| Setting | Default | Notes |
|---------|---------|-------|
| MODEL_SIZE | medium | tiny/base/small/medium/large-v2/large-v3 |
| DEVICE | auto | auto=detect GPU, "cuda"/"cpu" to force |
| COMPUTE_TYPE | auto | auto → int8_float16 on GPU (3.4x faster, same quality), int8 on CPU |
| SAMPLE_RATE | 16000 | whisper expects 16kHz |
| MIC_DEVICE | "Jabra" | substring match on device name, None=default |
| LANGUAGE | "ru" | "auto" = detect |
| BEAM_SIZE | 5 | beam search width (1 = fastest) |
| HOTKEY | f9 | push-to-talk key (hold) |
| CONTINUOUS_HOTKEY | f10 | continuous mode toggle |
| HISTORY_KEY | f8 | paste from history (repeated presses go further back) |
| HISTORY_SIZE | 10 | dictations kept in memory |
| SOUND_FEEDBACK | True | beep on record start (880 Hz) / stop (440 Hz) |
| OVERLAY_ENABLED | True | mic level bar bottom-right while recording |
| INITIAL_PROMPT | "Привет, как..." | Russian text hint for punctuation |
| VOCABULARY | ["Метраникс", ...] | domain terms appended to initial_prompt |
| LOG_FILE | %TEMP%\voicebutton.log | log file path |
| LOG_MAX_BYTES | 1048576 | 1 MB — log rotation threshold |
| LOG_BACKUP_COUNT | 3 | number of rotated backup files |
| VRAM_MIN_GB | {medium:5, large-v3:10, ...} | minimum VRAM per model (pre-load check via nvidia-smi) |
| MODEL_DOWNLOAD_GB | {medium:1.5, large-v3:3.0, ...} | approximate download size for first-run UI |
| MODEL_CACHE_DIR | None | None = HuggingFace default (~/.cache/huggingface/hub) |

## Releases

Two pre-built EXE variants are distributed via GitHub Releases:

| Build | MODEL_SIZE | VRAM | Target GPU |
|-------|-----------|------|------------|
| voicebutton-medium.exe | medium | 6 GB+ | RTX 2060/3060/4060 |
| voicebutton-large.exe | large-v3 | 10 GB+ | RTX 3080/4070/4080 |

Build both via `python build_releases.py` (patches MODEL_SIZE, runs PyInstaller, renames output, restores original).
