"""
SpoofCNN - the AI-vs-human voice classifier.

Deliberately small. With a hackathon-sized dataset a large network memorises the
specific voice generators in the training set and collapses on an unseen one.
Four conv blocks at ~65k parameters trains in minutes and exports cleanly to
TFLite for the on-device path.

The one non-obvious choice is the head: it averages over the TIME axis and keeps
the FREQUENCY axis intact before the classifier.

    global average pooling  ->  [B, 64]        loses which band was odd
    mean over time only     ->  [B, 64, F]     keeps it

Vocoder artifacts are frequency-localised - a particular band behaving in a way
a human vocal tract cannot produce. Collapsing frequency averages that evidence
away. Time is where invariance is wanted, because it should not matter *when*
inside the second the artifact shows up.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Label convention, matching asvspoof5_spectrograms.LABEL_MAP:
#   0 = bonafide (human), 1 = spoof (AI).
# The model emits ONE logit; sigmoid(logit) is P(spoof) = P(AI voice).
HUMAN, AI = 0, 1


class ConvBlock(nn.Module):
    """3x3 conv -> BatchNorm -> ReLU -> 2x2 max-pool."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpoofCNN(nn.Module):
    """
    Log spectrogram [B, 1, F, T] -> one logit [B].

    F is 257 for the full band or 104 with band_crop; pass whichever you trained
    on via n_freq_bins so the classifier is sized correctly.
    """

    CHANNELS = (16, 32, 64, 64)

    def __init__(self, n_freq_bins: int = 257, dropout: float = 0.3):
        super().__init__()
        blocks = []
        c_prev = 1
        for c in self.CHANNELS:
            blocks.append(ConvBlock(c_prev, c))
            c_prev = c
        self.features = nn.Sequential(*blocks)

        # Four 2x2 pools quarter the frequency axis twice over: 257 -> 16, 104 -> 6.
        f_out = n_freq_bins
        for _ in self.CHANNELS:
            f_out //= 2
        if f_out < 1:
            raise ValueError(
                f"n_freq_bins={n_freq_bins} is too small for "
                f"{len(self.CHANNELS)} pooling stages"
            )

        self.n_freq_bins = n_freq_bins
        self.flat_dim = c_prev * f_out
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.flat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:                      # [B, F, T] -> [B, 1, F, T]
            x = x.unsqueeze(1)
        x = self.features(x)                 # [B, 64, F', T']
        x = x.mean(dim=-1)                   # mean over TIME -> [B, 64, F']
        x = torch.flatten(x, 1)              # keep frequency -> [B, 64*F']
        x = self.dropout(x)
        return self.classifier(x).squeeze(-1)   # [B]

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """P(AI voice) in [0, 1]. Puts the module in eval mode and restores it."""
        was_training = self.training
        self.eval()
        p = torch.sigmoid(self(x))
        self.train(was_training)
        return p

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(band_crop: bool = False, **kwargs) -> SpoofCNN:
    """Construct a model matching the feature config in features.py."""
    from features import feature_shape

    _, n_freq_bins, _ = feature_shape(band_crop)
    return SpoofCNN(n_freq_bins=n_freq_bins, **kwargs)


if __name__ == "__main__":
    from features import feature_shape

    for crop in (False, True):
        shape = feature_shape(crop)
        m = build_model(band_crop=crop)
        x = torch.randn(4, *shape)
        out = m(x)
        print(
            f"band_crop={str(crop):5s} in={tuple(x.shape)} -> out={tuple(out.shape)}  "
            f"flat_dim={m.flat_dim:5d}  params={m.n_parameters():,}"
        )
