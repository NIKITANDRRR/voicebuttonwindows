"""Record 3 benchmark phrases from the default mic (same sounddevice path as the app).

Beep = start speaking. Durations: 7s / 8s / 18s.
"""
import sys
import time
import wave

import numpy as np
import sounddevice as sd

sys.stdout.reconfigure(encoding="utf-8")
SR = 16000

# Resolve mic exactly like the app: substring match, fallback default
MIC_DEVICE = "Jabra"
mic_dev = None
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0 and MIC_DEVICE.lower() in d["name"].lower():
        mic_dev = i
        break
if mic_dev is None:
    mic_dev = sd.default.device[0]
print(f"Mic: #{mic_dev} {sd.query_devices(mic_dev)['name']}")


def beep(freq=880, dur=0.25):
    import winsound
    winsound.Beep(freq, int(dur * 1000))


PHRASES = [
    ("bench_real_1.wav", 7, "Привет, отправь мне ТЭО по САКВ до пятницы, пожалуйста."),
    ("bench_real_2.wav", 8, "Метраникс внедрил систему на ЧЦЗ, КМ прошёл согласование, ТФВС в работе."),
    ("bench_real_3.wav", 18, "Мы подняли сервис, чтобы можно было с ноутбука заходить в локалке. Только там нельзя передать показания счётчиков, придётся видимо звонить каждый месяц."),
]

print("Default input device:", sd.query_devices(sd.default.device[0])["name"])

for fname, dur, text in PHRASES:
    print(f"\n>>> SAY: {text}")
    print(f"    ({dur}s window)")
    sys.stdout.flush()
    time.sleep(2.0)
    beep()
    beep()
    time.sleep(1.0)
    print("    recording...", flush=True)
    audio = sd.rec(int(SR * dur), samplerate=SR, channels=1, dtype="float32", device=mic_dev)
    sd.wait()
    peak = np.abs(audio).max()
    print(f"    peak={peak:.3f}")
    with wave.open(fname, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio.flatten() * 32767).astype(np.int16).tobytes())
    print(f"    saved {fname}")

print("\nDONE")
