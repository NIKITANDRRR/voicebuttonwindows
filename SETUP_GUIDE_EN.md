# VoiceButton — step-by-step setup

> This guide covers a clean setup: checking your GPU, installing the NVIDIA driver, picking the right build, and the first run. Work through the steps in order.
>
> Русская версия: [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

## What the program does

VoiceButton types what you say. Hold **F9**, speak, release **F9** — the text appears wherever the cursor is: Telegram, Word, a browser, an email.

With an NVIDIA graphics card recognition is fast. Without one the app still runs, just slower.

---

## What you need

| Item | Why |
|---|---|
| Windows 10 or 11 | the OS the app runs on |
| NVIDIA graphics card | speeds up recognition |
| NVIDIA driver | without it the app can't see the GPU |
| Microphone | you speak into it |

You do **not** need the full CUDA Toolkit (the multi-gigabyte developer package). The regular NVIDIA driver is enough — that's what people usually mean when they say "CUDA driver".

---

## Step 1. Check for an NVIDIA graphics card

1. Press **Win + X** (or right-click the Start button).
2. Open **Device Manager**.
3. Expand the **Display adapters** section.
4. Two possibilities:
   - **"NVIDIA GeForce …"** is listed — you have the card, go to step 2.
   - Only **"Intel …"** or **"AMD …"** — no NVIDIA card. The app will run on the CPU; skip to step 4.

---

## Step 2. Install the NVIDIA driver

The driver is the software that lets Windows talk to your graphics card. You install it once.

### Option A — via the NVIDIA App (automatic)

1. Download the NVIDIA App (or GeForce Experience) from `https://www.nvidia.com/`.
2. Install and open it.
3. Go to the **Drivers** tab → **Check for updates** → **Install**.
4. Restart the computer after it finishes.

### Option B — manually from the NVIDIA site

1. Open the driver search page: `https://www.nvidia.com/download/index.aspx`.
2. Pick your card series and model (e.g. GeForce RTX 3060), your OS (Windows 10/11, 64-bit), and language.
3. Click **Search**, download the file it offers, and run it.
4. Restart the computer.

---

## Step 3. Verify the driver

1. Press **Win + R**, type `cmd`, press Enter.
2. In the black window type `nvidia-smi` and press Enter.
3. A table with your GPU name and driver version means everything is installed.
4. `'nvidia-smi' is not recognized…` means the driver isn't there — back to step 2.

---

## Step 4. Download the app and pick a build

The [Releases](https://github.com/NIKITANDRRR/voicebuttonwindows/releases) page has two ready-made files. Choose by the amount of VRAM (video memory):

| File | VRAM needed | Typical GPUs |
|---|---|---|
| `voicebutton-medium.exe` | 6 GB+ | RTX 2060, 3060, 4060 |
| `voicebutton-large.exe` | 10 GB+ | RTX 3080, 4070, 4080 |

How to check your VRAM:

1. **Ctrl + Shift + Esc** opens Task Manager.
2. **Performance** tab → **GPU** on the left.
3. The **Dedicated GPU memory** line is your VRAM.

If in doubt, take `voicebutton-medium.exe`: it fits most modern cards, and its Russian and English accuracy is already high.

---

## Step 5. Run the app

> ⚠️ The app must run as Administrator — otherwise it can't track the F9 key and won't work.

1. Find the downloaded `.exe` file.
2. Right-click → **Run as administrator**.
3. If Windows SmartScreen asks whether to run: **More info** → **Run anyway**. That's a standard warning for apps outside the store.
4. If your antivirus blocks the file, allow it — the app's source code is open and lives in this repository.

To always run elevated: right-click the file → **Properties** → **Compatibility** tab → check **"Run this program as an administrator"** → **OK**.

---

## Step 6. First launch

The first launch downloads the recognition model: about **1.5 GB** for medium and **3 GB** for large. This happens once; after that the model is stored locally in `C:\Users\<name>\.cache\huggingface\hub\`.

A "Downloading Whisper … model" window shows the progress — wait for it to finish. Then a microphone icon appears in the tray (next to the clock): green means ready, red means recording.

The app has no window; it works from the tray. To quit, right-click the mic icon → **Exit**.

---

## Step 7. Check that it works

1. Open any editor (Notepad, Word) or a chat.
2. Put the cursor where you want the text.
3. Hold **F9** and speak.
4. Release **F9** — a second later the text lands at the cursor.

Two more things:

- Two quick F9 presses start continuous recording; the next F9 press stops it.
- Your clipboard is safe: the app saves it before pasting and restores it afterwards.

---

## Troubleshooting

### The tray says "CPU mode (no GPU)"

The app didn't find an NVIDIA card and is running on the CPU. Not an error, just slower. If you do have a card, check steps 2–3: the driver is probably missing or outdated.

### No text appears, log says "Audio is SILENT"

Recording goes to the wrong microphone. Check that the mic is connected and selected: **Settings → System → Sound → Input**. And that mic access is allowed: **Settings → Privacy & security → Microphone**.

### Windows or antivirus blocks the file

See step 5.

### Where are the logs

The log lives at `%TEMP%\voicebutton.log` — paste that path into File Explorer to open the file.

---

## Running from source

You'll need Python 3.12, the NVIDIA driver, and the CUDA 12 + cuDNN 9 libraries for GPU mode. Installation is covered in the [faster-whisper GPU section](https://github.com/SYSTRAN/faster-whisper#gpu).

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
