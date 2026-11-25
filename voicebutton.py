"""
VoiceButton — push-to-talk transcription via F9.

Hold F9 to record, release to transcribe and paste at cursor.
F10 toggles continuous mode — keeps recording until F10 pressed again.

Pastes result into active window at cursor position via Ctrl+V.
Runs as system tray icon (no console window).

Settings can be overridden per machine via voicebutton.ini next to
the script (or next to the exe) — see voicebutton.ini.example.
"""

import sys
import os
import time
import subprocess
import threading
import logging
import tempfile
from pathlib import Path
import numpy as np

# ── Config ──────────────────────────────────────────────
MODEL_SIZE = "medium"       # tiny/base/small/medium/large-v2/large-v3
DEVICE = "auto"             # "auto" (detect GPU), "cuda" or "cpu"
COMPUTE_TYPE = "auto"       # "auto" (float16 on GPU, int8 on CPU) or explicit
SAMPLE_RATE = 16000         # whisper expects 16kHz
MIC_DEVICE = "Jabra"        # None = default, or substring match on device name
LANGUAGE = "ru"             # "ru" = force Russian, "auto" = detect
BEAM_SIZE = 5               # beam search width
HOTKEY = "f9"               # push-to-talk key (hold to record)
CONTINUOUS_HOTKEY = "f10"   # continuous mode toggle (press once)
HISTORY_KEY = "f8"          # paste from history (repeated presses go further back)
HISTORY_SIZE = 10           # recent dictations kept in memory
SOUND_FEEDBACK = True       # beep on record start/stop
OVERLAY_ENABLED = True      # mic level bar overlay while recording

# Initial prompt gives Whisper a hint to add punctuation in Russian.
# Without it, Whisper often omits commas and periods in dictation.
INITIAL_PROMPT = "Привет, как дела? Это пример текста с запятыми, точками и вопросительными знаками."

# Domain terms the recognizer tends to mangle (company names, jargon).
# Appended to the initial prompt — fixes e.g. "Metronics" for "Метраникс".
VOCABULARY = ["Метраникс", "Метран", "ПГ Метран", "КМ", "ЧЗ", "ТФВС", "САКВ", "САК", "ФИО", "артикул"]

LOG_FILE = os.path.join(tempfile.gettempdir(), "voicebutton.log")
LOG_MAX_BYTES = 1_048_576  # 1 MB per file
LOG_BACKUP_COUNT = 3       # keep last 3 rotated files (4 MB total max)

# Model info — used for VRAM checks and first-run download messages.
# VRAM includes model weights + inference overhead (beam search, KV cache).
VRAM_MIN_GB = {
    "tiny": 1,
    "base": 1,
    "small": 2,
    "medium": 5,
    "large-v2": 10,
    "large-v3": 10,
}
# Approximate download size (ctranslate2 float16 format).
MODEL_DOWNLOAD_GB = {
    "tiny": 0.07,
    "base": 0.14,
    "small": 0.30,
    "medium": 1.5,
    "large-v2": 3.0,
    "large-v3": 3.0,
}
# Where to store downloaded models. None = HuggingFace default:
#   Windows: C:\Users\<user>\.cache\huggingface\hub\
#   Linux:   ~/.cache/huggingface/hub/
MODEL_CACHE_DIR = None
# ────────────────────────────────────────────────────────

# Logging (file-based since no console in tray mode).
# Rotating handler prevents log from filling up disk.
log = logging.getLogger("voicebutton")
log.setLevel(logging.INFO)
if LOG_FILE:
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(fh)

# ── External config file (voicebutton.ini) ───────────────
# Optional per-machine overrides, so repo code and installed copies
# never diverge: work laptop can run small/cpu while home PC runs
# medium/cuda — same code, different ini. See voicebutton.ini.example.
CONFIG_FILE = Path(
    sys.executable if getattr(sys, "frozen", False) else __file__
).with_suffix(".ini")


def _load_config_file():
    global MODEL_SIZE, DEVICE, COMPUTE_TYPE, BEAM_SIZE, LANGUAGE, MIC_DEVICE
    global HOTKEY, CONTINUOUS_HOTKEY, HISTORY_KEY, HISTORY_SIZE, INITIAL_PROMPT, VOCABULARY, MODEL_CACHE_DIR
    global SOUND_FEEDBACK, OVERLAY_ENABLED

    import configparser
    parser = configparser.ConfigParser()
    if not CONFIG_FILE.exists():
        return
    if not parser.read(CONFIG_FILE, encoding="utf-8"):
        log.warning(f"Config file unreadable: {CONFIG_FILE}")
        return
    log.info(f"Config loaded: {CONFIG_FILE}")

    MODEL_SIZE = parser.get("model", "size", fallback=MODEL_SIZE)
    DEVICE = parser.get("model", "device", fallback=DEVICE).lower()
    COMPUTE_TYPE = parser.get("model", "compute", fallback=COMPUTE_TYPE).lower()
    BEAM_SIZE = parser.getint("model", "beam", fallback=BEAM_SIZE)
    _lang = parser.get("model", "language", fallback=LANGUAGE).strip().lower()
    LANGUAGE = None if _lang in ("", "auto") else _lang

    _mic = parser.get("audio", "mic", fallback="").strip()
    if _mic:
        MIC_DEVICE = _mic

    HOTKEY = parser.get("hotkeys", "ptt", fallback=HOTKEY).strip().lower()
    CONTINUOUS_HOTKEY = parser.get("hotkeys", "continuous", fallback=CONTINUOUS_HOTKEY).strip().lower()
    HISTORY_KEY = parser.get("hotkeys", "history", fallback=HISTORY_KEY).strip().lower()
    HISTORY_SIZE = parser.getint("hotkeys", "history_size", fallback=HISTORY_SIZE)

    SOUND_FEEDBACK = parser.getboolean("feedback", "sound", fallback=SOUND_FEEDBACK)
    OVERLAY_ENABLED = parser.getboolean("feedback", "overlay", fallback=OVERLAY_ENABLED)

    _prompt = parser.get("prompt", "initial_prompt", fallback="").strip()
    if _prompt:
        INITIAL_PROMPT = _prompt
    _terms = parser.get("prompt", "terms", fallback="").strip()
    if _terms:
        VOCABULARY = [t.strip() for t in _terms.split(",") if t.strip()]

    _cache = parser.get("model", "cache_dir", fallback="").strip()
    if _cache:
        MODEL_CACHE_DIR = _cache


_load_config_file()


def build_initial_prompt():
    """Punctuation sample + domain terms, within Whisper's ~224-token prompt budget."""
    prompt = INITIAL_PROMPT.strip()
    if VOCABULARY:
        prompt += " Термины: " + ", ".join(VOCABULARY) + "."
    return prompt


# ── Helper functions ─────────────────────────────────────


def detect_device():
    """Auto-detect best device. Returns (device, compute_type).

    Checks if CUDA is available via ctranslate2 (the backend faster-whisper uses).
    Falls back to CPU if no NVIDIA driver/CUDA toolkit is installed.
    """
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count > 0:
            log.info(f"CUDA detected ({cuda_count} device(s)) — using GPU")
            return ("cuda", "float16")
        else:
            log.warning("No CUDA device found via ctranslate2 — falling back to CPU")
            return ("cpu", "int8")
    except Exception as e:
        log.warning(f"CUDA check failed ({e}) — falling back to CPU")
        return ("cpu", "int8")


def get_vram_mb():
    """Get GPU VRAM in MB via nvidia-smi. Returns None if unavailable."""
    try:
        kwargs = {"capture_output": True, "text": True, "timeout": 5}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            **kwargs,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def is_model_cached(model_size):
    """Check if the Whisper model is already downloaded."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return False
    pattern = f"faster-whisper-{model_size}"
    for d in cache_dir.iterdir():
        if pattern in d.name.lower():
            # Verify actual model files exist (not just a partial download)
            if list(d.rglob("model.bin")) or list(d.rglob("*.safetensors")):
                return True
    return False


def show_error_dialog(title, message):
    """Show a tkinter error dialog."""
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()


def show_loading_window(model_size, downloading):
    """Show a centered loading window with indeterminate progress bar.

    Returns the tk.Tk root. Caller destroys it when done.
    """
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("VoiceButton")
    root.geometry("420x170")
    root.resizable(False, False)

    # Center on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 420) // 2
    y = (root.winfo_screenheight() - 170) // 2
    root.geometry(f"+{x}+{y}")

    if downloading:
        dl_gb = MODEL_DOWNLOAD_GB.get(model_size, "?")
        line1 = f"Downloading Whisper {model_size} model (~{dl_gb} GB)"
        line2 = "First run only — please wait."
    else:
        line1 = f"Loading Whisper {model_size} model..."
        line2 = ""

    lbl1 = tk.Label(root, text=line1, font=("Segoe UI", 12, "bold"))
    lbl1.pack(pady=(30, 5))
    if line2:
        lbl2 = tk.Label(root, text=line2, font=("Segoe UI", 10), fg="gray")
        lbl2.pack(pady=(0, 15))
    else:
        lbl2 = tk.Label(root, text="", font=("Segoe UI", 10))
        lbl2.pack(pady=(0, 15))

    progress = ttk.Progressbar(root, mode="indeterminate", length=360)
    progress.pack(pady=(0, 20))
    progress.start(12)

    root.update()
    return root


def load_model_with_ui(model_size, device, compute_type, download_root):
    """Load Whisper model with VRAM check + loading window.

    Shows an error dialog and calls sys.exit(1) on failure.
    """
    # ── Step 1: VRAM check ──
    if device == "cuda":
        vram = get_vram_mb()
        required_gb = VRAM_MIN_GB.get(model_size, 0)
        if vram is not None and vram < required_gb * 1024:
            vram_gb = vram / 1024
            show_error_dialog(
                "VoiceButton — Not Enough VRAM",
                f"Whisper {model_size} requires ~{required_gb} GB VRAM.\n"
                f"Your GPU has {vram_gb:.0f} GB.\n\n"
                f"Please use voicebutton-medium.exe instead.",
            )
            log.error(f"VRAM check failed: {vram_gb:.1f} GB < {required_gb} GB required")
            sys.exit(1)
        elif vram is not None:
            log.info(f"VRAM check OK: {vram / 1024:.1f} GB available, {required_gb} GB required")
        else:
            log.warning("Could not query VRAM — proceeding without check")

    # ── Step 2: Check if model needs downloading ──
    downloading = not is_model_cached(model_size)

    # ── Step 3: Show loading window ──
    if downloading:
        log.info(f"First run — downloading model '{model_size}' (~{MODEL_DOWNLOAD_GB.get(model_size, '?')} GB)...")
    else:
        log.info(f"Loading cached model '{model_size}'...")

    win = show_loading_window(model_size, downloading)

    # ── Step 4: Load in background thread ──
    result = {"model": None, "error": None}

    def _load():
        try:
            kwargs = dict(device=device, compute_type=compute_type)
            if download_root:
                kwargs["download_root"] = download_root
            result["model"] = __import__("faster_whisper").WhisperModel(model_size, **kwargs)
        except Exception as e:
            result["error"] = e
            log.error(f"Model load error: {e}")
        finally:
            win.after(0, win.destroy)

    threading.Thread(target=_load, daemon=True).start()
    win.mainloop()  # blocks until window is destroyed

    # ── Step 5: Check result ──
    if result["error"]:
        err_str = str(result["error"])
        show_error_dialog(
            "VoiceButton — Error",
            f"Failed to load Whisper {model_size}:\n{err_str}",
        )
        sys.exit(1)

    log.info("Model loaded.")
    return result["model"]


def main():
    import ctypes
    import keyboard
    import sounddevice as sd
    import pystray
    from PIL import Image, ImageDraw

    # Check admin rights (keyboard hook needs it on Windows)
    if sys.platform == "win32":
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        if not is_admin:
            log.warning("Not running as Administrator — keyboard hook may fail")
        else:
            log.info("Running as Administrator — OK")

    # List audio devices
    log.info("Audio devices:")
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    log.info(f"  Default input: device #{default_in}")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            log.info(f"  #{i}: {d['name']} (in:{d['max_input_channels']})")

    # Resolve device/compute: "auto" → detect GPU, explicit config wins
    if DEVICE == "auto":
        device = detect_device()[0]
    else:
        device = DEVICE
        log.info(f"Device from config: {device}")
    if COMPUTE_TYPE == "auto":
        # int8_float16: 3.4x faster than float16 on GTX 16xx with equal quality
        # (bench 25.11: 18s speech -> 1.7s vs 5.5s, texts identical or better)
        compute_type = "int8_float16" if device == "cuda" else "int8"
    else:
        compute_type = COMPUTE_TYPE

    # Load model (with VRAM check + loading window)
    model = load_model_with_ui(MODEL_SIZE, device, compute_type, MODEL_CACHE_DIR)

    # Warm up so the first real dictation is not slow (first transcribe
    # triggers CUDA kernel init / lazy loading — used to cost ~10s).
    try:
        t0 = time.time()
        model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32),
                         language=LANGUAGE, beam_size=1)
        log.info(f"Model warmed up in {time.time() - t0:.1f}s")
    except Exception as e:
        log.warning(f"Warmup failed: {e}")

    # Show notification if running on CPU (no NVIDIA driver)
    if device == "cpu":
        cpu_nag = True
    else:
        cpu_nag = False

    # Resolve mic device
    mic_dev = MIC_DEVICE
    if mic_dev is not None and isinstance(mic_dev, str):
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and mic_dev.lower() in d["name"].lower():
                mic_dev = i
                log.info(f"Mic matched: #{i} {sd.query_devices(i)['name']}")
                break
        else:
            log.warning(f"Mic '{MIC_DEVICE}' not found, using default")

    # ── State ───────────────────────────────────────────────
    recording = {"active": False, "frames": [], "lock": threading.Lock()}
    # mode: "idle" | "ptt" (push-to-talk, waiting for release) | "continuous"
    state = {"mode": "idle"}
    # Last N dictations: items (newest last), index = how far back HISTORY_KEY went
    history = {"items": [], "index": 0}
    # Mic level overlay: level fed by the audio callback, visible flag by start/stop
    overlay = {"level": 0.0, "visible": False}
    icon_ref = {"icon": None}  # filled after icon creation

    # ── Tray icon ───────────────────────────────────────────
    def mic_icon(is_recording=False):
        """Draw a simple microphone icon. Red when recording, green when idle."""
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        color = (255, 80, 80, 255) if is_recording else (100, 200, 100, 255)
        # Mic body (rounded rectangle)
        draw.rounded_rectangle([24, 8, 40, 32], radius=8, fill=color)
        # Arc stand
        draw.arc([16, 20, 48, 50], start=0, end=180, fill=color, width=3)
        # Stem
        draw.line([32, 48, 32, 56], fill=color, width=3)
        # Base
        draw.line([24, 56, 40, 56], fill=color, width=3)
        return img

    def set_tray(recording_active, text):
        if icon_ref["icon"]:
            icon_ref["icon"].icon = mic_icon(recording_active)
            icon_ref["icon"].title = f"VoiceButton — {text}"

    # ── Audio ───────────────────────────────────────────────
    def audio_callback(indata, frames, time_info, status):
        if recording["active"]:
            recording["frames"].append(indata.copy())
            # feed the level bar (portaudio thread: only cheap math here)
            block = indata[:, 0]
            overlay["level"] = float(np.sqrt(np.mean(block ** 2)))

    def play_beep(freq, dur=0.12):
        """Non-blocking beep; a worker thread so winsound never stalls the hook."""
        if not SOUND_FEEDBACK:
            return

        def _b():
            try:
                import winsound
                winsound.Beep(freq, int(dur * 1000))
            except Exception:
                pass

        threading.Thread(target=_b, daemon=True).start()

    def start_recording():
        recording["active"] = True
        recording["frames"] = []
        overlay["visible"] = True
        play_beep(880, 0.10)
        set_tray(True, "Recording...")
        log.info("● Recording...")

    def stop_and_transcribe():
        recording["active"] = False
        overlay["visible"] = False
        play_beep(440, 0.15)
        set_tray(False, "Transcribing...")
        log.info("Transcribing...")

        with recording["lock"]:
            if not recording["frames"]:
                set_tray(False, "Ready")
                return
            audio = np.concatenate(recording["frames"], axis=0)
            recording["frames"] = []

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32).flatten()

        duration = len(audio) / SAMPLE_RATE
        peak = np.abs(audio).max()
        rms = np.sqrt(np.mean(audio ** 2))
        log.info(f"dur={duration:.1f}s peak={peak:.4f} rms={rms:.4f}")
        if duration < 0.3:
            log.info("Too short, ignoring.")
            set_tray(False, "Ready")
            return
        if peak < 0.01:
            log.warning("Audio is SILENT — wrong mic or muted?")
            set_tray(False, "Ready")
            return

        segments, info = model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            vad_filter=True,
            initial_prompt=build_initial_prompt(),
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        log.info(f">> {text}")

        if text:
            history["items"].append(text)
            if len(history["items"]) > HISTORY_SIZE:
                history["items"] = history["items"][-HISTORY_SIZE:]
            history["index"] = 0
            update_history_menu()
            paste_text(text)
        else:
            log.info("(no speech detected)")
        set_tray(False, "Ready")

    def paste_text(text):
        """Paste text at cursor position. Saves/restores clipboard."""
        import pyperclip

        try:
            saved = pyperclip.paste()
        except Exception:
            saved = None

        pyperclip.copy(text)
        time.sleep(0.05)

        try:
            keyboard.press("ctrl")
            keyboard.press("v")
            keyboard.release("v")
            keyboard.release("ctrl")
            log.info("Ctrl+V sent.")
        except Exception as e:
            log.error(f"keyboard Ctrl+V failed: {e}")
            try:
                import pyautogui
                pyautogui.hotkey("ctrl", "v")
            except Exception as e2:
                log.error(f"pyautogui fallback failed: {e2}")

        def _restore():
            time.sleep(0.3)
            try:
                if saved is not None:
                    pyperclip.copy(saved)
            except Exception:
                pass

        threading.Thread(target=_restore, daemon=True).start()

    # ── Hotkey handlers ─────────────────────────────────────
    def on_press(event):
        # Continuous mode → stop and transcribe
        if state["mode"] == "continuous":
            stop_and_transcribe()
            state["mode"] = "idle"
            return

        # Normal push-to-talk start
        if state["mode"] == "idle":
            start_recording()
            state["mode"] = "ptt"

    def on_release(event):
        # Release = transcribe immediately (no double-press wait)
        if state["mode"] != "ptt":
            return
        state["mode"] = "idle"
        stop_and_transcribe()

    def on_continuous(event):
        # CONTINUOUS_HOTKEY toggles continuous: record until pressed again
        if state["mode"] == "continuous":
            stop_and_transcribe()
            state["mode"] = "idle"
        else:
            if state["mode"] == "idle":
                start_recording()
            state["mode"] = "continuous"
            log.info("Continuous mode activated.")

    def on_history(event):
        # HISTORY_KEY re-pastes dictations; each further press goes one step back
        items = history["items"]
        if not items:
            log.info("History is empty.")
            return
        i = history["index"] % len(items)
        text = items[len(items) - 1 - i]
        history["index"] = i + 1
        log.info(f"History paste ({i + 1}/{len(items)} back): {text[:60]}")
        paste_text(text)

    # ── Mic level overlay ───────────────────────────────────
    # Thin always-on-top bar bottom-right: green grows with voice level,
    # red on clipping. Fed by audio_callback via overlay["level"].
    def overlay_loop():
        import tkinter as tk

        W, H = 160, 12
        root = tk.Tk()
        root.overrideredirect(True)  # no title bar/border
        root.attributes("-topmost", True)
        canvas = tk.Canvas(root, width=W, height=H, bg="#1c1c1c", highlightthickness=0)
        canvas.pack()
        bar = canvas.create_rectangle(0, 0, 0, H, fill="#39d353", width=0)

        root.update_idletasks()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{W}x{H}+{sw - W - 24}+{sh - H - 96}")

        shown = {"v": False}

        def tick():
            # speech rms ~0.05-0.15 -> scale x8 so normal voice fills the bar
            lvl = min(1.0, overlay["level"] * 8.0)
            w = int(W * lvl)
            canvas.coords(bar, 0, 0, w, H)
            canvas.itemconfigure(bar, fill="#f85149" if lvl >= 0.98 else "#39d353")
            if overlay["visible"] and not shown["v"]:
                root.deiconify()
                root.attributes("-topmost", True)
                shown["v"] = True
            elif not overlay["visible"] and shown["v"]:
                root.withdraw()
                shown["v"] = False
            root.after(50, tick)

        root.withdraw()
        tick()
        root.mainloop()

    def _run_overlay():
        try:
            overlay_loop()
        except Exception as e:
            log.warning(f"Overlay failed (app continues without it): {e}")

    if OVERLAY_ENABLED:
        threading.Thread(target=_run_overlay, daemon=True).start()

    # ── Start audio stream ──────────────────────────────────
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1),
        device=mic_dev,
    )
    stream.start()

    # Bind hotkeys
    keyboard.on_press_key(HOTKEY, on_press, suppress=False)
    keyboard.on_release_key(HOTKEY, on_release, suppress=False)
    keyboard.on_press_key(CONTINUOUS_HOTKEY, on_continuous, suppress=False)
    keyboard.on_press_key(HISTORY_KEY, on_history, suppress=False)

    log.info(f"Ready. Hold {HOTKEY.upper()} to record, {CONTINUOUS_HOTKEY.upper()} toggles continuous, "
             f"{HISTORY_KEY.upper()} pastes history.")

    # ── Tray menu ───────────────────────────────────────────
    # Find model cache path for info display
    cache_display = MODEL_CACHE_DIR or str(Path.home() / ".cache" / "huggingface" / "hub")

    def on_exit(icon, item):
        log.info("Exiting...")
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        keyboard.unhook_all()
        icon.stop()

    def _make_history_paster(idx):
        def _paste(icon, item):
            if idx < len(history["items"]):
                paste_text(history["items"][idx])
        return _paste

    def build_menu():
        items = [
            pystray.MenuItem("VoiceButton", None, enabled=False),
            pystray.MenuItem(f"{HOTKEY.upper()} hold = talk, {CONTINUOUS_HOTKEY.upper()} = continuous", None, enabled=False),
            pystray.MenuItem(f"{HISTORY_KEY.upper()} = paste from history ({HISTORY_SIZE} last)", None, enabled=False),
            pystray.Menu.SEPARATOR,
        ]
        if history["items"]:
            sub = []
            for pos, txt in enumerate(reversed(history["items"])):
                label = txt if len(txt) <= 50 else txt[:50] + "..."
                sub.append(pystray.MenuItem(label, _make_history_paster(len(history["items"]) - 1 - pos)))
            items.append(pystray.Menu("History", *sub))
            items.append(pystray.Menu.SEPARATOR)
        items += [
            pystray.MenuItem(f"Model: Whisper {MODEL_SIZE} ({device})", None, enabled=False),
            pystray.MenuItem(f"Cache: {cache_display}", None, enabled=False),
            pystray.MenuItem(f"Log: {LOG_FILE}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", on_exit),
        ]
        return pystray.Menu(*items)

    def update_history_menu():
        if icon_ref["icon"]:
            try:
                icon_ref["icon"].menu = build_menu()
            except Exception as e:
                log.warning(f"History menu update failed: {e}")

    tooltip = "VoiceButton — Ready"
    if cpu_nag:
        tooltip = "VoiceButton — CPU mode (no GPU)"

    icon = pystray.Icon(
        "VoiceButton",
        mic_icon(False),
        tooltip,
        menu=build_menu(),
    )
    icon_ref["icon"] = icon

    icon.run()  # blocks — main loop


if __name__ == "__main__":
    main()
