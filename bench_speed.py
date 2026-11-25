"""Benchmark: beam_size x compute_type on the home GPU (GTX 1660 SUPER).

Variants: float16/int8_float16 x beam 5/1, model medium, VAD + app prompt,
language ru — exactly the app's transcribe() call path.
"""
import time
import sys
import wave

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import voicebutton as vb
from faster_whisper import WhisperModel

PROMPT = vb.build_initial_prompt()
LANG = vb.LANGUAGE
VAD_PARAMS = dict(min_silence_duration_ms=300, speech_pad_ms=200)


def load_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


FILES = [
    ("bench_real_1.wav", 7, "Привет, отправь мне ТЭО по САКВ до пятницы, пожалуйста."),
    ("bench_real_2.wav", 8, "Метраникс внедрил систему на ЧЦЗ, КМ прошёл согласование, ТФВС в работе."),
    ("bench_real_3.wav", 18, "Мы подняли сервис, чтобы можно было с ноутбука заходить в локалке. Только там нельзя передать показания счётчиков, придётся видимо звонить каждый месяц."),
]
AUDIO = {name: load_wav(name) for name, _, _ in FILES}

VARIANTS = [
    ("float16     beam5 (current)", "float16", 5),
    ("float16     beam1", "float16", 1),
    ("int8_float16 beam5", "int8_float16", 5),
    ("int8_float16 beam1", "int8_float16", 1),
]

results = {}
for compute in ("float16", "int8_float16"):
    t0 = time.time()
    model = WhisperModel("medium", device="cuda", compute_type=compute)
    print(f"\n=== loaded medium/{compute} in {time.time()-t0:.1f}s ===")
    # warmup
    model.transcribe(np.zeros(16000, dtype=np.float32), language=LANG, beam_size=1)

    for label, comp, beam in [v for v in VARIANTS if v[1] == compute]:
        print(f"\n--- {label} ---")
        for name, ref_dur, ref_text in FILES:
            audio = AUDIO[name]
            t0 = time.time()
            segs, _ = model.transcribe(
                audio, language=LANG, beam_size=beam, vad_filter=True,
                initial_prompt=PROMPT, vad_parameters=VAD_PARAMS,
            )
            text = " ".join(s.text.strip() for s in segs).strip()
            dt = time.time() - t0
            rtf = dt / ref_dur
            results.setdefault(label, []).append((name, ref_dur, dt, rtf, text))
            print(f"{name}: dur={ref_dur:5.1f}s  time={dt:5.2f}s  x{rtf:.2f} of speech")
            print(f"    text: {text}")

print("\n\n=========== SUMMARY ===========")
print(f"{'variant':30s} {'short':>12s} {'terms':>12s} {'long':>12s} {'avg x realtime':>15s}")
for label, rows in results.items():
    times = [f"{r[2]:.2f}s" for r in rows]
    avg_rtf = sum(r[3] for r in rows) / len(rows)
    print(f"{label:30s} {times[0]:>12s} {times[1]:>12s} {times[2]:>12s} {avg_rtf:>14.2f}x")
print("\nDone.")
