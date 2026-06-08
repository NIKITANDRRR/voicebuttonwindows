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
import threading
import logging
import tempfile
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
# ────────────────────────────────────────────────────────

# Logging (file-based since no console in tray mode)
log = logging.getLogger("voicebutton")
log.setLevel(logging.INFO)
if LOG_FILE:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(fh)


def main():
    import ctypes
    import keyboard
    import sounddevice as sd
    import pystray
    from PIL import Image, ImageDraw
    from faster_whisper import WhisperModel

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

    # Load model
    log.info(f"Loading model '{MODEL_SIZE}' on {DEVICE}...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    log.info("Model loaded.")

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
            pystray.MenuItem("Exit", on_exit),
        ),
    )
    icon_ref["icon"] = icon

    icon.run()  # blocks — main loop


if __name__ == "__main__":
    main()
