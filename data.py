"""
Dataset wiring: augmentation, generator holdout, class balancing.

The holdout is the important part. ASVspoof5 rows carry an attack_id naming the
system that produced the spoof (A09, A10, ...). Training on five generators and
testing on the same five measures memorisation, not detection - the model learns
those five, and a sixth walks straight past it. Splitting BY attack_id puts an
entirely unseen generator in the test set. That number is lower, and it is the
only one worth quoting.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from asvspoof5_spectrograms import ASVspoof5Dataset
from features import LogLinearSpectrogram
from training.augment import SpecAugment, WaveformAugment


class AugmentedSpoofDataset(Dataset):
    """
    Wraps ASVspoof5Dataset to augment in the WAVEFORM domain.

    The base dataset is asked for waveforms rather than spectrograms so the
    phone-channel simulation happens before the STFT - degrading audio and then
    transforming is the real signal path; degrading a finished spectrogram is not.
    """

    def __init__(
        self,
        base: ASVspoof5Dataset,
        indices: list[int] | None = None,
        wave_aug: WaveformAugment | None = None,
        spec_aug: SpecAugment | None = None,
        band_crop: bool = False,
    ):
        if base.return_type not in ("waveform", "both"):
            raise ValueError(
                "base dataset must return waveforms; construct it with "
                "return_type='waveform'"
            )
        self.base = base
        self.indices = list(range(len(base))) if indices is None else list(indices)
        self.wave_aug = wave_aug
        self.spec_aug = spec_aug
        self.extractor = LogLinearSpectrogram(band_crop=band_crop)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        sample = self.base[self.indices[i]]
        waveform = sample["waveform"]

        if self.wave_aug is not None:
            waveform = self.wave_aug(waveform)

        spec = self.extractor(waveform)
        if self.spec_aug is not None:
            spec = self.spec_aug(spec)

        return {
            "spectrogram": spec,
            "label": sample["label"].float(),
            "attack_id": sample["attack_id"],
            "file_id": sample["file_id"],
        }

    def labels(self) -> list[int]:
        return [self.base.records[j].label_idx for j in self.indices]


def holdout_split(
    base: ASVspoof5Dataset,
    holdout_attacks: list[str],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split indices into (train, val, test) where test contains ONLY the held-out
    generators plus enough bonafide rows to stay measurable.

    Bonafide audio has no attack_id, so it is split by fraction instead - the
    holdout is about unseen *generators*, and holding out human speech too would
    just shrink the data.
    """
    holdout = set(holdout_attacks)
    rng = random.Random(seed)

    test, seen, bonafide = [], [], []
    for i, rec in enumerate(base.records):
        if rec.label == "bonafide":
            bonafide.append(i)
        elif rec.attack_id in holdout:
            test.append(i)
        else:
            seen.append(i)

    rng.shuffle(bonafide)
    n_test_bona = max(1, int(len(bonafide) * 0.2)) if bonafide else 0
    test += bonafide[:n_test_bona]
    remaining = bonafide[n_test_bona:] + seen
    rng.shuffle(remaining)

    n_val = int(len(remaining) * val_fraction)
    return remaining[n_val:], remaining[:n_val], test


def balanced_sampler(labels: list[int]) -> WeightedRandomSampler:
    """
    ASVspoof protocols run roughly 10:1 spoof:bonafide. Without rebalancing the
    model can score ~90% by answering "spoof" every time.
    """
    counts = torch.bincount(torch.tensor(labels), minlength=2).float().clamp(min=1)
    weights = (1.0 / counts)[torch.tensor(labels)]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def build_loaders(
    protocol_path: str | Path,
    audio_dir: str | Path,
    holdout_attacks: list[str],
    batch_size: int = 32,
    band_crop: bool = False,
    augment: bool = True,
    num_workers: int = 0,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Build train/val/test loaders. Augmentation is applied to train only."""
    base = ASVspoof5Dataset(
        protocol_path=protocol_path,
        audio_dir=audio_dir,
        return_type="waveform",
        skip_missing=True,
    )
    tr_idx, va_idx, te_idx = holdout_split(base, holdout_attacks, seed=seed)

    train_ds = AugmentedSpoofDataset(
        base, tr_idx,
        wave_aug=WaveformAugment(enabled=augment),
        spec_aug=SpecAugment(enabled=augment),
        band_crop=band_crop,
    )
    # No augmentation on val/test - they measure the real channel, not a
    # randomised one, and a moving target is not a metric.
    val_ds = AugmentedSpoofDataset(base, va_idx, band_crop=band_crop)
    test_ds = AugmentedSpoofDataset(base, te_idx, band_crop=band_crop)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=balanced_sampler(train_ds.labels()),
        num_workers=num_workers, drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    info = {
        "n_train": len(tr_idx),
        "n_val": len(va_idx),
        "n_test": len(te_idx),
        "holdout_attacks": sorted(set(holdout_attacks)),
        "test_attacks": sorted({base.records[i].attack_id for i in te_idx
                                if base.records[i].attack_id}),
    }
    return train_loader, val_loader, test_loader, info
