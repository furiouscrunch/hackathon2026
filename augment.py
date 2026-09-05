"""
Augmentation. This is where the accuracy actually comes from.

Two families:

  Waveform - push clean training audio through the same channel the live audio
  suffers: 300-3400 Hz bandpass, compression, 8 kHz resample, G.711 mu-law.
  Same chain as phone_call_effect.py, but in pure torch rather than shelling out
  to ffmpeg per clip, because this runs on every sample of every epoch.

  Spectrogram - SpecAugment frequency/time masking, applied after features.py.

Train on clean audio and deploy on phone audio and the model fails. This is not
an optimisation.
"""

from __future__ import annotations

import random

import torch
import torchaudio.functional as AF

from features import SAMPLE_RATE

TELEPHONE_RATE = 8000
HIGHPASS_HZ = 300.0
LOWPASS_HZ = 3400.0


def phone_channel(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    mulaw: bool = True,
) -> torch.Tensor:
    """
    Clean audio -> telephone audio. Mirrors phone_call_effect.FILTER_CHAIN.

    Bandpass first, so nothing survives above the 4 kHz Nyquist to alias when we
    drop to 8 kHz. Returns at the original sample rate with the damage baked in.
    """
    x = waveform
    if x.ndim == 1:
        x = x.unsqueeze(0)

    x = AF.highpass_biquad(x, sample_rate, HIGHPASS_HZ)
    x = AF.lowpass_biquad(x, sample_rate, LOWPASS_HZ)
    x = _compress(x)

    x = AF.resample(x, sample_rate, TELEPHONE_RATE)
    if mulaw:
        # Real 8-bit logarithmic companding, the actual source of G.711's grain.
        q = AF.mu_law_encoding(x.clamp(-1.0, 1.0), quantization_channels=256)
        x = AF.mu_law_decoding(q, quantization_channels=256)
    x = AF.resample(x, TELEPHONE_RATE, sample_rate)

    return x


def _compress(x: torch.Tensor, threshold: float = 0.125, ratio: float = 3.0) -> torch.Tensor:
    """
    Static-curve stand-in for ffmpeg's acompressor (-18 dB, 3:1).

    No attack/release envelope - a per-sample curve is enough to reproduce the
    flattened dynamics that phone AGC produces, and it stays differentiable and
    fast.
    """
    mag = x.abs()
    over = (mag - threshold).clamp(min=0.0)
    target = threshold + over / ratio
    gain = torch.where(mag > threshold, target / mag.clamp(min=1e-8), torch.ones_like(mag))
    return x * gain


def room_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add white noise at a given SNR. Stands in for speakerphone room pickup."""
    sig_pow = waveform.pow(2).mean().clamp(min=1e-12)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    return waveform + torch.randn_like(waveform) * noise_pow.sqrt()


def gain_jitter(waveform: torch.Tensor, db_range: tuple[float, float]) -> torch.Tensor:
    """Random level shift. Recording levels vary wildly between handsets."""
    db = random.uniform(*db_range)
    return waveform * (10 ** (db / 20))


class WaveformAugment:
    """
    Randomised channel simulation, applied to the waveform before features.py.

    phone_prob below 1.0 keeps some clean audio in the mix so the model does not
    come to rely on codec artifacts as the only cue.
    """

    def __init__(
        self,
        phone_prob: float = 0.8,
        noise_prob: float = 0.5,
        snr_db_range: tuple[float, float] = (10.0, 30.0),
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        enabled: bool = True,
    ):
        self.phone_prob = phone_prob
        self.noise_prob = noise_prob
        self.snr_db_range = snr_db_range
        self.gain_db_range = gain_db_range
        self.enabled = enabled

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return waveform
        x = waveform
        if random.random() < self.phone_prob:
            x = phone_channel(x)
        if random.random() < self.noise_prob:
            x = room_noise(x, random.uniform(*self.snr_db_range))
        x = gain_jitter(x, self.gain_db_range)
        return x.clamp(-1.0, 1.0)


class SpecAugment:
    """
    Frequency and time masking, applied after features.py.

    Masks are filled with the spectrogram's own mean, which is ~0 post
    normalisation, so a mask reads as "no information" rather than as a loud edge.
    """

    def __init__(
        self,
        freq_masks: int = 2,
        freq_width: int = 16,
        time_masks: int = 2,
        time_width: int = 8,
        enabled: bool = True,
    ):
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width = time_width
        self.enabled = enabled

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return spec
        out = spec.clone()
        fill = out.mean()
        n_freq, n_time = out.shape[-2], out.shape[-1]

        for _ in range(self.freq_masks):
            w = random.randint(0, min(self.freq_width, n_freq - 1))
            if w:
                f0 = random.randint(0, n_freq - w)
                out[..., f0 : f0 + w, :] = fill
        for _ in range(self.time_masks):
            w = random.randint(0, min(self.time_width, n_time - 1))
            if w:
                t0 = random.randint(0, n_time - w)
                out[..., :, t0 : t0 + w] = fill
        return out
