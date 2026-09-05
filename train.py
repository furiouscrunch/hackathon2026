"""
Train SpoofCNN to tell an AI voice from a human one.

    python -m training.train ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv \
        --audio-dir audio/flac_D --holdout-attack A09 --epochs 15

Reports per-window accuracy and a 6-second smoothed accuracy, because that is
what the app actually shows. Single-window scores are noisy; the deployed
verdict averages twelve of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from features import feature_shape
from live_capture import SMOOTHING_WINDOWS
from training.data import build_loaders
from training.model import build_model

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def evaluate(model: nn.Module, loader, device: str) -> dict:
    """Per-window metrics plus a smoothed proxy and the spoof-side recall."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["spectrogram"].to(device)
            probs.append(torch.sigmoid(model(x)).cpu())
            labels.append(batch["label"])
    if not probs:
        return {"n": 0}

    p = torch.cat(probs)
    y = torch.cat(labels)
    pred = (p > 0.5).float()

    acc = (pred == y).float().mean().item()
    ai, human = y == 1, y == 0
    return {
        "n": int(y.numel()),
        "acc": acc,
        "recall_ai": (pred[ai] == 1).float().mean().item() if ai.any() else float("nan"),
        "recall_human": (pred[human] == 0).float().mean().item() if human.any() else float("nan"),
        "acc_smoothed": _smoothed_accuracy(p, y),
    }


def _smoothed_accuracy(p: torch.Tensor, y: torch.Tensor, k: int = SMOOTHING_WINDOWS) -> float:
    """
    Accuracy after averaging every k consecutive window scores.

    An approximation - it groups whatever order the loader produced rather than
    true consecutive windows from one call - but it shows the direction smoothing
    moves the number, which is the point.
    """
    n = (p.numel() // k) * k
    if n == 0:
        return float("nan")
    pm = p[:n].view(-1, k).mean(dim=1)
    ym = y[:n].view(-1, k).mean(dim=1)
    keep = (ym == 0) | (ym == 1)          # only groups with a single true label
    if not keep.any():
        return float("nan")
    return (((pm[keep] > 0.5).float()) == ym[keep]).float().mean().item()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="training.train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("protocol", help="Path to an ASVspoof5 Track-1 .tsv")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--holdout-attack", action="append", default=[],
                    help="Attack ID kept out of training entirely. Repeatable. "
                         "Without at least one, the reported number is memorisation.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--band-crop", action="store_true",
                    help="Restrict features to the telephone band -> [1, 104, 63]")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(CHECKPOINT_DIR / "spoofcnn.pt"))
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(a.protocol).is_file():
        print(f"Protocol file not found: {a.protocol}")
        return 1
    if not a.holdout_attack:
        print("WARNING: no --holdout-attack given. Test accuracy will measure "
              "memorisation of the generators in your training set, not detection "
              "of unseen ones. Pass --holdout-attack A09 (or similar).\n")

    train_loader, val_loader, test_loader, info = build_loaders(
        protocol_path=a.protocol,
        audio_dir=a.audio_dir,
        holdout_attacks=a.holdout_attack,
        batch_size=a.batch_size,
        band_crop=a.band_crop,
        augment=not a.no_augment,
        num_workers=a.num_workers,
        seed=a.seed,
    )
    print(f"feature shape {feature_shape(a.band_crop)}  device={device}")
    print(f"train={info['n_train']}  val={info['n_val']}  test={info['n_test']}")
    print(f"held out: {info['holdout_attacks'] or 'NOTHING'}   "
          f"attacks in test: {info['test_attacks']}\n")
    if info["n_train"] == 0:
        print("No training rows. Check --audio-dir actually contains the .flac files.")
        return 1

    model = build_model(band_crop=a.band_crop).to(device)
    print(f"SpoofCNN: {model.n_parameters():,} parameters\n")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -1.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, a.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            x = batch["spectrogram"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * y.numel()
            seen += y.numel()
        sched.step()

        val = evaluate(model, val_loader, device)
        train_loss = total / max(seen, 1)
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}  "
              f"val acc {val.get('acc', float('nan')):.3f}  "
              f"(AI {val.get('recall_ai', float('nan')):.3f} / "
              f"human {val.get('recall_human', float('nan')):.3f})")

        if val.get("acc", -1) > best_val:
            best_val = val["acc"]
            torch.save({
                "state_dict": model.state_dict(),
                "n_freq_bins": model.n_freq_bins,
                "band_crop": a.band_crop,
                "feature_shape": list(feature_shape(a.band_crop)),
                "epoch": epoch,
                "val_acc": best_val,
                "holdout_attacks": info["holdout_attacks"],
            }, a.out)

    print(f"\nbest val acc {best_val:.3f} -> {a.out}")

    test = evaluate(model, test_loader, device)
    print("\n--- held-out generator test ---")
    if test.get("n"):
        print(json.dumps(test, indent=2))
        print("\nThis is the number to quote. Same-generator accuracy is memorisation.")
    else:
        print("Test split is empty - no rows matched the held-out attack IDs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
