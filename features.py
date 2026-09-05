"""
Canonical feature extraction for the voice-scam detector.

SINGLE SOURCE OF TRUTH. Training (asvspoof5_spectrograms.py) and live capture
(live_capture.py) both import from here, so they cannot drift apart. When the
Kotlin on-device spectrogram is written, it must reproduce THESE numbers
exactly - see PARITY below.

Feature spec
------------
    sample rate   16000 Hz, mono
    window        1.0 s  = 16000 samples (matches CallAudioCapture.WINDOW_SAMPLES)
    n_fft         512    -> 257 frequency bins (n_fft // 2 + 1)
    hop_length    256
    win_length    512, Hann (periodic)
    center        True, reflect padding -> 63 frames (1 + 16000 // 256)
    magnitude     power=1.0 (LINEAR magnitude, NOT mel, NOT power)
    log           natural log(magnitude + 1e-6)
    normalise     per-window: subtract mean, divide by std over the whole 2-D
                  spectrogram (not per-band, not per-frame)

    output shape  [1, 257, 63]

Linear, not mel: mel spacing compresses the high frequencies, and that is
exactly where vocoder/TTS artifacts concentrate. Mel would average away the
evidence the model is supposed to key on.

PARITY
------
The Kotlin implementation must match on every one of: sample rate, n_fft, hop,
window function AND its periodic/symmetric flag, centering + pad mode, whether
magnitude or power is taken, the log epsilon, and the ORDER of log vs
normalise (log FIRST, then normalise). A mismatch in any one of them feeds the
deployed model inputs unlike anything it trained on, while training metrics
still look perfect.

To check parity once the Kotlin exists, run:
    python features.py --emit-reference
which writes a reference WAV plus the expected tensor as CSV. Run the same WAV
through the Kotlin path and diff - they should agree to ~3 decimal places.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchaudio.transforms import Spectrogram

# ---------------------------------------------------------------- constants

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = 16000          # CallAudioCapture.WINDOW_SAMPLES
HOP_SECONDS = 0.5
HOP_WINDOW_SAMPLES = 8000       # CallAudioCapture.HOP_SAMPLES

N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512

N_FREQ_BINS = N_FFT // 2 + 1                    # 257
N_FRAMES = 1 + WINDOW_SAMPLES // HOP_LENGTH     # 63
FEATURE_SHAPE = (1, N_FREQ_BINS, N_FRAMES)      # [1, 257, 63]

LOG_EPS = 1e-6
NORM_EPS = 1e-5

# Dynamic-range floor, in natural-log units below the window's own peak.
# 80 dB = 80 / (20 * log10(e)) = 9.21 nats.
TOP_DB = 80.0
TOP_LOG = 9.21

# Telephone passband in bin indices, for the optional --band-crop.
# Bin width is SAMPLE_RATE / N_FFT = 31.25 Hz, so 300 Hz -> bin 10 and
# 3400 Hz -> bin 109. Everything above is destroyed by the phone codec long
# before it reaches the mic; see the README on why this is worth trying.
BAND_CROP_LO = 8
BAND_CROP_HI = 112
N_CROPPED_BINS = BAND_CROP_HI - BAND_CROP_LO    # 104
CROPPED_SHAPE = (1, N_CROPPED_BINS, N_FRAMES)   # [1, 104, 63]


def feature_shape(band_crop: bool = False) -> tuple[int, int, int]:
    """Output shape for the given config. Use this instead of hardcoding."""
    return CROPPED_SHAPE if band_crop else FEATURE_SHAPE


class LogLinearSpectrogram:
    """
    Waveform [1, T] -> log linear-magnitude spectrogram [1, 257, frames].

    For the standard 1 s window the output is [1, 257, 63]. Frozen defaults -
    change them here and BOTH training and inference change together, which is
    the point of this module.

    band_crop keeps only the telephone passband, giving [1, 104, frames].
    Off by default so the wire format stays fixed; turn it on in BOTH training
    and inference together or not at all.
    """

    def __init__(
        self,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        log_eps: float = LOG_EPS,
        normalise: bool = True,
        top_db: float | None = TOP_DB,
        band_crop: bool = False,
    ):
        self.log_eps = log_eps
        self.normalise = normalise
        self.top_db = top_db
        self.band_crop = band_crop
        self.spec = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=torch.hann_window,   # periodic Hann; Kotlin must match
            power=1.0,                     # magnitude, not power
            center=True,
            pad_mode="reflect",
            normalized=False,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mag = self.spec(waveform)
        out = torch.log(mag + self.log_eps)

        if self.band_crop:
            out = out[..., BAND_CROP_LO:BAND_CROP_HI, :]

        if self.top_db is not None:
            # Clamp to a fixed dynamic range below this window's own peak.
            # Without this, standardisation stretches the near-silent band
            # above 3.4 kHz - which the phone codec already emptied - into
            # full-scale noise, burying the speech detail underneath it.
            floor = out.max() - (self.top_db / TOP_DB) * TOP_LOG
            out = torch.clamp(out, min=float(floor))

        if self.normalise:
            # Per-window standardisation. Costs a little on clean data, but
            # keeps wildly varying phone recording levels from shifting the
            # input distribution at inference time. Applied AFTER the log and
            # AFTER the floor - the Kotlin port must use this same order.
            out = (out - out.mean()) / (out.std() + NORM_EPS)
        return out


def fix_length(waveform: torch.Tensor, n_samples: int = WINDOW_SAMPLES) -> torch.Tensor:
    """Centre-crop or zero-pad a [1, T] waveform to exactly n_samples."""
    n = waveform.shape[-1]
    if n > n_samples:
        start = (n - n_samples) // 2
        return waveform[:, start : start + n_samples]
    if n < n_samples:
        return torch.nn.functional.pad(waveform, (0, n_samples - n))
    return waveform


def spectrogram_from_window(
    window,
    extractor: "LogLinearSpectrogram | None" = None,
    band_crop: bool = False,
) -> torch.Tensor:
    """
    One live capture window -> model-ready tensor [1, 257, 63].

    `window` is anything tensor-like holding mono float samples in [-1, 1] at
    16 kHz - the FloatArray CallAudioCapture emits, decoded on this side.
    Short or long input is centre-cropped/padded rather than rejected, so a
    ragged final chunk still scores.
    """
    if not isinstance(window, torch.Tensor):
        window = torch.as_tensor(window, dtype=torch.float32)
    window = window.to(torch.float32)
    if window.ndim == 1:
        window = window.unsqueeze(0)
    elif window.ndim == 2 and window.shape[0] > 1:
        window = window.mean(dim=0, keepdim=True)   # mono-ise defensively
    window = fix_length(window)
    extractor = extractor or LogLinearSpectrogram(band_crop=band_crop)
    return extractor(window)


def _emit_reference(out_dir: Path) -> None:
    """Write a reference WAV + expected tensor, for Python<->Kotlin diffing."""
    import csv

    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    # Deterministic, broadband, non-trivial: a sweep plus fixed-seed noise, so
    # a window-function or centering mismatch actually shows up in the diff.
    t = torch.arange(WINDOW_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
    sweep = torch.sin(2 * torch.pi * (200 + 3000 * t) * t)
    torch.manual_seed(0)
    sig = (0.7 * sweep + 0.1 * torch.randn(WINDOW_SAMPLES)).clamp(-1.0, 1.0)

    wav_path = out_dir / "parity_reference.wav"
    sf.write(str(wav_path), sig.numpy(), SAMPLE_RATE, subtype="PCM_16")

    spec = spectrogram_from_window(sig).squeeze(0)
    csv_path = out_dir / "parity_reference.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in spec.tolist():
            w.writerow([f"{v:.6f}" for v in row])

    print(f"wrote {wav_path}")
    print(f"wrote {csv_path}  shape={tuple(spec.shape)} (freq_bins x frames)")
    print("Run the same WAV through the Kotlin spectrogram and diff to ~3dp.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Canonical feature extractor")
    p.add_argument("--emit-reference", action="store_true",
                   help="Write parity_reference.wav + .csv for Kotlin cross-checking")
    p.add_argument("--out-dir", default="parity", help="Where to write reference files")
    a = p.parse_args()

    print(f"sample_rate={SAMPLE_RATE} window={WINDOW_SAMPLES} n_fft={N_FFT} "
          f"hop={HOP_LENGTH} win={WIN_LENGTH}")
    print(f"feature shape = {FEATURE_SHAPE}")
    if a.emit_reference:
        _emit_reference(Path(a.out_dir))
