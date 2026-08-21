# 📘 VoiceButton — Step-by-Step Setup Guide for Beginners

> If you're not a programmer or sysadmin and just want the app to type what you say — this guide is for you. Everything is explained step by step. You don't need to memorize anything — just follow along in order.

---

## What this program does

**VoiceButton** types what you say. Hold the **F9** key, speak into your microphone, release **F9** — and the text appears wherever your cursor is (Telegram, Word, browser, email — anywhere).

It works fast if you have an **NVIDIA** graphics card. Without one it still works, just slower (see below).

---

## What you need (short version)

| What | Why |
|---|---|
| Windows 10 or 11 | the system itself |
| An NVIDIA graphics card | so speech recognition runs fast |
| NVIDIA driver (a.k.a. the "CUDA driver") | so the app can talk to the GPU |
| A microphone | obvious :) |

**Important:** you do **NOT** need to install the full "CUDA Toolkit" (a huge multi-gigabyte developer package). For the ready-made app, the regular **NVIDIA driver** is enough — that's what people often call the "CUDA driver" in casual talk.

---

## Step 1. Check whether you have an NVIDIA graphics card

1. Press **Win + X** (or right-click the Start button).
2. Select **Device Manager**.
3. Expand the **Display adapters** section.
4. Look at what's listed:
   - You see **"NVIDIA GeForce …"** — great, you have the card. Go to Step 2.
   - You only see **"Intel …"** or **"AMD …"** — there's no NVIDIA card. The app will still work (on the CPU), just slower. You can jump straight to Step 4.

---

## Step 2. Install (or update) the NVIDIA driver

The driver is the software that teaches Windows how to use your graphics card. You install it once.

### Option A (easier, automatic) — via the NVIDIA App / GeForce Experience

1. Go to `https://www.nvidia.com/geforce-experience/` (or the NVIDIA App page) and download **GeForce Experience** / **NVIDIA App**.
2. Install it (just keep clicking Next).
3. Open it → **Drivers** tab → **Check for updates** → **Install**.
4. Wait for it to finish and restart your computer.

### Option B (manually, no extra software)

1. Open `https://www.nvidia.com/download/index.aspx`.
2. Select your graphics card (series and model — e.g. GeForce RTX 3060), your OS (Windows 10/11, 64-bit), and language.
3. Click **Search** → download the driver → run the downloaded file → click Next until it's done.
4. Restart your computer.

---

## Step 3. Verify the driver installed correctly

1. Press **Win + R**, type `cmd`, press Enter.
2. In the black window type `nvidia-smi` and press Enter.
3. If you see a table with your GPU name and driver version — everything is fine.
4. If it says `'nvidia-smi' is not recognized as an internal or external command` — the driver didn't install; go back to Step 2.

---

## Step 4. Download the app and pick the right version

The [Releases](https://github.com/NIKITANDRRR/voicebuttonwindows/releases) page has two ready-made files. Which one to pick depends on your GPU's **VRAM** (video memory):

| File | VRAM required | Typical GPUs |
|---|---|---|
| `voicebutton-medium.exe` | ~6 GB+ | RTX 2060, 3060, 4060 and similar |
| `voicebutton-large.exe` | ~10 GB+ | RTX 3080, 4070, 4080 and similar |

**How to check your VRAM:**

1. Press **Ctrl + Shift + Esc** (Task Manager opens).
2. Go to the **Performance** tab → click **GPU** on the left.
3. Look at the **Dedicated GPU memory** line — that's your VRAM in GB.

Not sure? Take `voicebutton-medium.exe`. It works on most modern GPUs, and its accuracy for Russian and English is already very good.

Download the file anywhere you like (e.g. your Desktop).

---

## Step 5. Run the app

> ⚠️ **Important:** the app must be run **as Administrator** — otherwise it can't capture the F9 key and simply won't work.

1. Find the downloaded `.exe` file.
2. **Right-click** it → **Run as administrator**.
3. If Windows SmartScreen asks "Run?" — click **"More info"** → **"Run anyway"** (normal for apps not from the store).
4. If your antivirus complains — allow the file (it's a common false positive; the app's source code is open and lives in this repository).

To avoid right-clicking every time, set it up once: right-click the file → **Properties** → **Compatibility** tab → check **"Run this program as an administrator"** → **OK**.

---

## Step 6. First launch (the model downloads)

On first launch the app downloads the speech recognition "model":

- medium — about **1.5 GB**;
- large — about **3 GB**.

This happens only once. A small "Downloading Whisper … model" window appears — just wait. The model is saved to `C:\Users\<your-name>\.cache\huggingface\hub\`, and next time the app starts instantly.

After loading, a microphone icon appears in the bottom-right corner (system tray, next to the clock): green — ready, red — recording.

> The app doesn't open a window — it lives in the tray. To close it, right-click the mic icon → **Exit**.

---

## Step 7. Check that everything works

1. Open any text editor (Notepad, Word) or a chat.
2. Place the cursor where you want the text.
3. **Hold F9** and speak into the microphone.
4. **Release F9** — a second later the text appears at the cursor.

**Extras:**

- **Two quick F9 presses** (press-release-press) start continuous mode — the app writes everything until you press F9 again.
- Your clipboard is safe — the app saves and restores whatever was in it before pasting.

---

## Troubleshooting

### The tray says "CPU mode (no GPU)"

The app didn't detect an NVIDIA card and switched to the CPU. Not an error — just slower.

- If you DO have an NVIDIA card — re-check Steps 2–3 (driver missing or outdated).
- If you don't have one — that's expected; everything runs on the CPU.

### No text appears / log says "Audio is SILENT"

The app is recording from the wrong microphone.

- Check the mic is connected and enabled: **Settings → System → Sound → Input**, pick the right microphone and make sure its volume isn't zero.
- Check mic access: **Settings → Privacy & security → Microphone** → allow access.

### Windows or antivirus blocks the file

See Step 5 — use "More info → Run anyway" and add the file to your antivirus exclusions.

### Where are the logs (to show someone an error)

The app writes a log to `%TEMP%\voicebutton.log`. Paste that path into File Explorer to open the file.

---

## For advanced users: running from source

If you want to run from source code (instead of the ready EXE), you'll need a bit more:

1. Install Python 3.12.
2. Install the NVIDIA driver (Step 2 above) **plus** the CUDA 12 and cuDNN 9 libraries — needed for GPU mode from source. Official instructions: [faster-whisper → GPU](https://github.com/SYSTRAN/faster-whisper#gpu).

```bash
git clone https://github.com/NIKITANDRRR/voicebuttonwindows.git
cd voicebuttonwindows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python voicebutton.py
```

Run as Administrator here too.

---

## License

MIT — see the [LICENSE](LICENSE) file.
