"""
VoiceButton — push-to-talk transcription via F9.

Hold F9 to record, release to transcribe and paste at cursor.
Double-click F9 (quick press-release-press) for continuous mode —
keeps recording until next F9 press.

Pastes result into active window at cursor position via Ctrl+V.
Runs as system tray icon (no console window).
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
DEVICE = "cuda"             # "cuda" or "cpu"
COMPUTE_TYPE = "float16"    # float16 for GPU, int8 for CPU
SAMPLE_RATE = 16000         # whisper expects 16kHz
MIC_DEVICE = "Jabra"        # None = default, or substring match on device name
LANGUAGE = None             # None = auto-detect, "ru" = force Russian
BEAM_SIZE = 5               # beam search width
HOTKEY = "f9"               # trigger key
DOUBLE_PRESS_THRESHOLD = 0.35  # seconds — max gap for double-click detection

# Initial prompt gives Whisper a hint to add punctuation in Russian.
# Without it, Whisper often omits commas and periods in dictation.
INITIAL_PROMPT = "Привет, как дела? Это пример текста с запятыми, точками и вопросительными знаками."

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


# ── Helper functions ─────────────────────────────────────

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

    # Load model (with VRAM check + loading window)
    model = load_model_with_ui(MODEL_SIZE, DEVICE, COMPUTE_TYPE, MODEL_CACHE_DIR)

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
    state = {"mode": "idle", "release_timer": None}
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

    def start_recording():
        recording["active"] = True
        recording["frames"] = []
        set_tray(True, "Recording...")
        log.info("● Recording...")

    def stop_and_transcribe():
        recording["active"] = False
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
            initial_prompt=INITIAL_PROMPT,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        log.info(f">> {text}")

        if text:
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

        # If a release-timer is pending → this is a double-click → go continuous
        if state["release_timer"]:
            state["release_timer"].cancel()
            state["release_timer"] = None
            state["mode"] = "continuous"
            log.info("Continuous mode activated.")
            return

        # Normal push-to-talk start
        if state["mode"] == "idle":
            start_recording()
            state["mode"] = "ptt"

    def on_release(event):
        if state["mode"] != "ptt":
            return  # continuous mode ignores release

        # Delay transcription — if another press comes within threshold,
        # on_press will cancel this timer and switch to continuous.
        def delayed():
            state["mode"] = "idle"
            state["release_timer"] = None
            stop_and_transcribe()

        t = threading.Timer(DOUBLE_PRESS_THRESHOLD, delayed)
        state["release_timer"] = t
        t.start()

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

    # Bind hotkey
    keyboard.on_press_key(HOTKEY, on_press, suppress=False)
    keyboard.on_release_key(HOTKEY, on_release, suppress=False)

    log.info(f"Ready. Hold {HOTKEY.upper()} to record, double-click for continuous.")

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

    icon = pystray.Icon(
        "VoiceButton",
        mic_icon(False),
        "VoiceButton — Ready",
        menu=pystray.Menu(
            pystray.MenuItem("VoiceButton", None, enabled=False),
            pystray.MenuItem("F9 hold = talk, 2x = continuous", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(f"Model: Whisper {MODEL_SIZE}", None, enabled=False),
            pystray.MenuItem(f"Cache: {cache_display}", None, enabled=False),
            pystray.MenuItem(f"Log: {LOG_FILE}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", on_exit),
        ),
    )
    icon_ref["icon"] = icon

    icon.run()  # blocks — main loop


if __name__ == "__main__":
    main()
