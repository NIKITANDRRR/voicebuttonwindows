"""
VoiceButton — push-to-talk transcription via Right Alt.
Records mic while Right Alt is held, transcribes with faster-whisper (CUDA),
types result into the active window.
"""

import sys
import time
import threading
import numpy as np

# ── Config ──────────────────────────────────────────────
MODEL_SIZE = "medium"       # whisper model: tiny/base/small/medium/large-v2/large-v3
DEVICE = "cuda"             # "cuda" or "cpu"
COMPUTE_TYPE = "float16"    # float16 for GPU, int8 for CPU
SAMPLE_RATE = 16000         # whisper expects 16kHz
MIC_DEVICE = "Jabra"       # None = default, or device number/name string
LANGUAGE = None             # None = auto-detect, "ru" = force Russian
BEAM_SIZE = 5               # beam search width
HOTKEY = "f12"              # trigger key
# ────────────────────────────────────────────────────────


def main():
    import ctypes
    import keyboard
    import sounddevice as sd
    from faster_whisper import WhisperModel

    # Check admin rights (keyboard hook needs it on Windows)
    if sys.platform == "win32":
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        if not is_admin:
            print("[VoiceButton] WARNING: Not running as Administrator!")
            print("[VoiceButton] Keyboard hook may not work. Re-run as admin.")
        else:
            print("[VoiceButton] Running as Administrator — OK")

    # List audio devices to help debug mic issues
    print("[VoiceButton] Audio devices:")
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    print(f"  Default input: device #{default_in}")
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            marker = " <-- DEFAULT" if i == default_in else ""
            print(f"  #{i}: {d['name']} (in:{d['max_input_channels']}){marker}")

    # Load model
    print(f"[VoiceButton] Loading model '{MODEL_SIZE}' on {DEVICE}...")
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    print(f"[VoiceButton] Model loaded. Press and hold [{HOTKEY}] to record.")

    # Recording state
    recording = {"active": False, "frames": [], "lock": threading.Lock()}

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if recording["active"]:
            recording["frames"].append(indata.copy())

    def on_press(event):
        if not recording["active"]:
            recording["active"] = True
            recording["frames"] = []
            print("[VoiceButton] ● Recording...", end="", flush=True)

    def on_release(event):
        if recording["active"]:
            recording["active"] = False
            print(" done.")

            with recording["lock"]:
                if not recording["frames"]:
                    return
                # Concatenate all frames into single array
                audio = np.concatenate(recording["frames"], axis=0)
                recording["frames"] = []

            # Convert to float32 mono if needed
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32).flatten()

            duration = len(audio) / SAMPLE_RATE
            peak = np.abs(audio).max()
            rms = np.sqrt(np.mean(audio ** 2))
            print(f" dur={duration:.1f}s peak={peak:.4f} rms={rms:.4f}")
            if duration < 0.3:
                print("[VoiceButton] Too short, ignoring.")
                return
            if peak < 0.01:
                print("[VoiceButton] Audio is SILENT — wrong mic or muted?")
                return

            print(f"[VoiceButton] Transcribing {duration:.1f}s...", end="", flush=True)
            t0 = time.time()

            segments, info = model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
            )

            # Collect text
            text = " ".join(seg.text.strip() for seg in segments).strip()
            elapsed = time.time() - t0
            print(f" ({elapsed:.1f}s)")

            if text:
                print(f"[VoiceButton] >> {text}")
                type_text(text)
            else:
                print("[VoiceButton] (no speech detected)")

    def type_text(text):
        """Type text into the currently active window."""
        import pyautogui
        import pyperclip

        # Use clipboard for instant paste — much faster than char-by-char typing
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")

    # Resolve mic device
    mic_dev = MIC_DEVICE
    if mic_dev is not None and isinstance(mic_dev, str):
        # Find by substring match
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0 and mic_dev.lower() in d['name'].lower():
                mic_dev = i
                print(f"[VoiceButton] Mic matched: #{i} {sd.query_devices(i)['name']}")
                break
        else:
            print(f"[VoiceButton] WARNING: Mic '{MIC_DEVICE}' not found, using default")

    # Start mic stream (always listening, audio_callback only saves when recording)
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1),  # 100ms chunks
        device=mic_dev,
    )
    stream.start()

    # Bind hotkey
    keyboard.on_press_key(HOTKEY, on_press, suppress=False)
    keyboard.on_release_key(HOTKEY, on_release, suppress=False)

    print(f"[VoiceButton] Ready. Hold [{HOTKEY}] to record, release to transcribe.")
    print(f"[VoiceButton] Press Ctrl+C in this window to exit.")

    # Keep alive
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[VoiceButton] Exiting.")


if __name__ == "__main__":
    main()
