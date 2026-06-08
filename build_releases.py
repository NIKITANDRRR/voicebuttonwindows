"""
Build both medium and large releases.
Outputs:
  dist/voicebutton-medium.exe  (Whisper medium,  ~6 GB VRAM)
  dist/voicebutton-large.exe   (Whisper large-v3, ~10 GB VRAM)
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SRC = ROOT / "voicebutton.py"

BUILDS = [
    ("medium",    "voicebutton-medium.exe"),
    ("large-v3",  "voicebutton-large.exe"),
]


def set_model_size(model: str) -> str:
    """Patch MODEL_SIZE in voicebutton.py, return previous value."""
    text = SRC.read_text(encoding="utf-8")
    m = re.search(r'^MODEL_SIZE\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("MODEL_SIZE not found in voicebutton.py")
    old = m.group(1)
    text = re.sub(
        r'^MODEL_SIZE\s*=\s*"[^"]+"',
        f'MODEL_SIZE = "{model}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    SRC.write_text(text, encoding="utf-8")
    return old


def build(out_name: str):
    print(f"\n=== Building {out_name} ===")
    subprocess.check_call(
        [str(PY), "-m", "PyInstaller", "voicebutton.spec", "--noconfirm"],
        cwd=str(ROOT),
    )
    built = ROOT / "dist" / "voicebutton.exe"
    target = ROOT / "dist" / out_name
    shutil.move(str(built), str(target))
    size_mb = target.stat().st_size / 1048576
    print(f"Done: {target}  ({size_mb:.0f} MB)")


def main():
    if not PY.exists():
        print(f"ERROR: Python not found at {PY}")
        sys.exit(1)

    original = set_model_size(BUILDS[0][0])  # save original value
    try:
        for model, out_name in BUILDS:
            set_model_size(model)
            build(out_name)
    finally:
        set_model_size(original)  # restore
    print(f"\nAll builds complete in dist/  (restored MODEL_SIZE={original})")


if __name__ == "__main__":
    main()
