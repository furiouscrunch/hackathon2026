# Voice Anti-Spoofing & Telecom Audio Pipeline

An end-to-end pipeline for AI voice scam detection, covering:
1. **Audio Preprocessing (`phone_call_effect.py`)**: Degrades high-fidelity speech (e.g., Indian accent recordings) into realistic telephony audio (G.711 $\mu$-law codec, 300–3,400 Hz bandpass, 8 kHz resampling).
2. **Spectrogram Generation (`asvspoof5_spectrograms.py`)**: Parses ASVspoof5 Track-1 metadata and extracts 80-bin log-mel spectrograms formatted for CNN model training.

---

## 1. Environment & Dependencies Setup

### Prerequisites
- **macOS** (Apple Silicon supported) or **Windows 10/11**
- **FFmpeg**: Required for audio filtering and codec transformations. Must be on `PATH`.
- **Python Virtual Environment (`.venv`)**: Python 3.9–3.14.

### Setup Commands (macOS / Linux)

```bash
# 1. Install FFmpeg (system package)
brew install ffmpeg          # macOS
# sudo apt install ffmpeg    # Debian/Ubuntu

# 2. Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install required Python packages
pip install -r requirements.txt
```

### Setup Commands (Windows — PowerShell)

```powershell
# 1. Install FFmpeg (adds it to PATH automatically)
winget install --id Gyan.FFmpeg -e
#    Alternatives: `choco install ffmpeg`, or download from
#    https://ffmpeg.org/download.html and add the bin\ folder to PATH.
#    Open a NEW terminal afterwards, then verify:
ffmpeg -version

# 2. Create and activate Python virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1
#    If activation is blocked by execution policy, run once:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. Install required Python packages
pip install -r requirements.txt
```

> On Windows, use `python` instead of `python3` in the commands below, and
> activate the venv with `.\.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.
> Multi-line commands shown with a trailing `\` need a backtick `` ` `` in PowerShell, or just write them on one line.

---

## 2. Phase 1: Audio Preprocessing (Phone Call Quality Simulator)

Scam and impersonation calls (banking fraud, delivery scams) typically reach victims over PSTN, VoIP, or cellular lines. To train and evaluate anti-spoofing models under realistic conditions, clean audio must be degraded to match real telephony acoustic channels.

### The DSP Degradation Chain in `phone_call_effect.py`

1. **Bandpass Filter (300 Hz – 3,400 Hz)**:
   - Traditional telephone networks strictly pass only 300–3,400 Hz.
   - Eliminates low-end chest rumble (< 300 Hz) and high-end air/sibilance (> 3.4 kHz), producing the classic "boxed-in, tinny" timbre.
2. **Dynamic Range Compression (`acompressor`)**:
   - `threshold=-18dB`, `ratio=3`, `attack=5ms`, `release=50ms`.
   - Simulates the Automatic Gain Control (AGC) and limiter hardware in phone handsets and PBX switchboards.
3. **8 kHz Resampling (`aresample=8000`)**:
   - Resamples audio to the telecommunications standard 8 kHz clock (Nyquist frequency 4 kHz).
4. **ITU-T G.711 $\mu$-law Codec Encoding (`pcm_mulaw`)**:
   - Passes audio through real 8-bit logarithmic companding (standard for US/Japan telecom and VoIP/SIP trunks).
   - Converted back to 16-bit PCM for universal player/ML compatibility with codec distortion permanently baked in.

### Commands to Run

#### Batch Process an Entire Audio Folder:
Process all `.mp3`, `.wav`, and `.flac` files in `audio/` (such as Indian accent recordings):

```bash
python3 phone_call_effect.py audio -o audio_phone_call
```

#### Process a Single File:
```bash
python3 phone_call_effect.py "audio/Demo_Podcast_1.mp3" -o audio_phone_call
```

#### Outputs Generated per Clip in `audio_phone_call/`:
- `<filename>_bandlimited.wav`: Bandpass filtered (300–3,400 Hz) + compressed + 8 kHz PCM.
- `<filename>_mulaw.wav`: The authentic G.711 $\mu$-law codec version (closest to real telephone calls).

---

## 3. Phase 2: Spectrogram Feature Extraction (`asvspoof5_spectrograms.py`)

Extracts normalized log-mel spectrograms from audio matched with ASVspoof5 Track-1 metadata.

### Acoustic & Tensor Specifications
- **Sample Rate**: 16,000 Hz (mono)
- **Duration**: Fixed 4.0 seconds center-cropped or padded (`max_duration_s=4.0`) so tensors stack cleanly into mini-batches
- **FFT Parameters**: 512-point FFT, 400-sample window (25 ms), 160-sample hop (10 ms)
- **Mel Scale**: 80 mel frequency bins spanning 20 Hz – 8,000 Hz (HTK scale, Slaney norm)
- **Log Compression**: Power-to-dB with an 80 dB clamp
- **Tensor Output Shape**: `[1, 80, 401]`

### Commands to Run

Make sure your virtual environment is active:
```bash
source .venv/bin/activate
```

#### Quick Test Export (16 spectrograms):
```bash
python asvspoof5_spectrograms.py ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv \
    --audio-dir audio/flac_D \
    --out-dir spectrograms \
    --export-n 16
```

#### Export with PyTorch `.pt` Tensors (for CNN training):
```bash
python asvspoof5_spectrograms.py ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv \
    --audio-dir audio/flac_D \
    --out-dir spectrograms \
    --export-n 16 \
    --save-pt
```

#### Export Only Tensors (skip PNG images):
```bash
python asvspoof5_spectrograms.py ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv \
    --audio-dir audio/flac_D \
    --out-dir spectrograms \
    --export-n 100 \
    --save-pt \
    --no-png
```

#### Full Split Export (All matched files in split):
```bash
python asvspoof5_spectrograms.py ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv \
    --audio-dir audio/flac_D \
    --out-dir spectrograms \
    --export-n -1
```

---

## 4. Python API Usage

You can also import and use the dataset and functions directly in Python training scripts:

```python
from torch.utils.data import DataLoader
from asvspoof5_spectrograms import ASVspoof5Dataset, parse_protocol_file, summarize_split

# 1. Inspect protocol balance
records = parse_protocol_file("ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv")
summarize_split(records, name="dev.track_1")

# 2. Load PyTorch Dataset
dataset = ASVspoof5Dataset(
    protocol_path="ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv",
    audio_dir="audio/flac_D",
    sample_rate=16000,
    max_duration_s=4.0,
    return_type="spectrogram"
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
batch = next(iter(loader))

print("Batch spectrogram shape:", batch["spectrogram"].shape)  # [32, 1, 80, 401]
print("Batch labels shape:", batch["label"].shape)              # [32] (0=bonafide, 1=spoof)
```

---

## 5. Engineering & Technical Solutions Implemented

1. **Python 3.14 Torchaudio Decoder Fix (macOS & Windows)**:
   - In Torchaudio 2.11+ under Python 3.13/3.14, `torchaudio.load()` routes through `torchcodec`, which fails when FFmpeg's shared libraries are missing (`libavutil` on macOS, the FFmpeg DLLs on Windows).
   - **Solution**: `_decode_audio()` wraps `torchaudio.load()` and falls back to `soundfile` (which bundles its own `libsndfile`), so audio decodes into PyTorch tensors with no system-level library dependency.
2. **FFmpeg Resampling Engine Compatibility**:
   - Homebrew's FFmpeg on macOS does not bundle `libsoxr` by default, causing `:resampler=soxr` to fail.
   - **Solution**: Updated the filter chain to `aresample=8000`, using FFmpeg's native `libswresample` engine.
3. **Directory Batch Processing**:
   - Enhanced `phone_call_effect.py` to accept directory paths and automatically process all immediate audio files while avoiding heavy subfolders (e.g. `audio/flac_D/`).

---

## 6. Directory Structure

```
hackathon/
├── README.md                     # This consolidated guide
├── requirements.txt              # Python dependencies
├── phone_call_effect.py          # Telephony audio degradation CLI tool
├── asvspoof5_spectrograms.py     # Protocol parser & log-mel spectrogram generator (CLI)
├── asvspoof5_dataset.py          # Library-only copy of the same parser/Dataset
├── .venv/                        # Local virtual environment (not committed)
├── my-folder/                    # Clean source audio (demo .mp3 / .wav recordings)
├── audio/                        # Expected location for ASVspoof5 audio
│   └── flac_D/                   # ASVspoof5 development set (.flac files)
├── audio_phone_call/             # Output folder for phone-degraded audio
├── ASVspoof5_protocols/          # Official ASVspoof5 protocol .tsv metadata
└── spectrograms/                 # Output folder for generated spectrograms (.png / .pt)
```

> The demo recordings currently live in `my-folder/`, not `audio/`. Either rename the
> folder or pass the real path, e.g. `python phone_call_effect.py my-folder -o audio_phone_call`.
