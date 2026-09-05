"""
Turn raw source audio into CNN-ready training data, reproducibly.

    python preprocess.py                       # my-folder/ -> processed/ + spectrograms/
    python preprocess.py --src my-folder --clean

Pipeline per source file:

    raw  ->  phone degradation  ->  1 s windows @ 0.5 s hop  ->  log spectrogram
             (phone_call_effect)     (drop silent windows)       (features.py)

Writes:
    processed/<name>_mulaw.wav      degraded audio, one per source
    windows/<name>/win_####.pt      [1, 257, 63] tensors, one per window
    spectrograms/<name>_####.png    the same windows as viewable images
    manifest.csv                    path, label, speaker, source_file, window_idx

LABELS: this script writes label=UNKNOWN for everything. It cannot tell an AI
voice from a human one - that is the thing being trained. Set the labels in
manifest.csv (or pass --label-map) before training, or training/train.py will
refuse to run.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from features import (
    HOP_WINDOW_SAMPLES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    LogLinearSpectrogram,
)
from phone_call_effect import FILTER_CHAIN, require_ffmpeg

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
# Windows quieter than this (RMS in [-1,1] units) are near-silence. Keeping them
# would mostly teach the model what a gap between sentences looks like.
SILENCE_RMS = 0.005


def degrade(src: Path, dst: Path) -> None:
    """Raw audio -> 8 kHz G.711 mu-law telephone audio, back to 16 kHz PCM."""
    tmp = dst.with_suffix(".mulaw.tmp.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-af", FILTER_CHAIN,
         "-ar", "8000", "-c:a", "pcm_mulaw", str(tmp)],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp), "-ar", str(SAMPLE_RATE),
         "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True, check=True,
    )
    tmp.unlink(missing_ok=True)


def load_mono(path: Path) -> torch.Tensor:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    w = torch.from_numpy(data).mean(dim=1)
    if sr != SAMPLE_RATE:
        import torchaudio
        w = torchaudio.functional.resample(w.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)
    return w


def save_png(spec: torch.Tensor, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = spec.squeeze(0).numpy()
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=110)
    ax.imshow(s, origin="lower", aspect="auto", cmap="magma",
              extent=[0, 1.0, 0, SAMPLE_RATE / 2000])
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("kHz", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.axhline(3.4, color="cyan", lw=0.7, ls="--", alpha=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="my-folder", help="Folder of raw source audio")
    ap.add_argument("--processed-dir", default="processed_audio")
    ap.add_argument("--windows-dir", default="windows")
    ap.add_argument("--png-dir", default="spectrograms")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--clean", action="store_true",
                    help="Wipe output dirs first (does NOT touch --src)")
    ap.add_argument("--png-per-file", type=int, default=6,
                    help="How many window images to save per source file (0 = none)")
    ap.add_argument("--band-crop", action="store_true")
    a = ap.parse_args(argv)

    require_ffmpeg()

    src_dir = Path(a.src)
    if not src_dir.is_dir():
        print(f"source folder not found: {src_dir}")
        return 1
    sources = sorted(f for f in src_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
    if not sources:
        print(f"no audio files in {src_dir}")
        return 1

    proc_dir, win_dir, png_dir = Path(a.processed_dir), Path(a.windows_dir), Path(a.png_dir)
    if a.clean:
        for d in (proc_dir, win_dir, png_dir):
            if d.exists():
                shutil.rmtree(d)
    for d in (proc_dir, win_dir, png_dir):
        d.mkdir(parents=True, exist_ok=True)

    extractor = LogLinearSpectrogram(band_crop=a.band_crop)
    rows = []
    total_win = total_dropped = 0

    print(f"{len(sources)} source file(s) from {src_dir}\n")
    for src in sources:
        stem = src.stem
        wav = proc_dir / f"{stem}_mulaw.wav"
        try:
            degrade(src, wav)
        except subprocess.CalledProcessError as e:
            print(f"  !! ffmpeg failed on {src.name}\n{e.stderr[-400:]}")
            continue

        audio = load_mono(wav)
        out_dir = win_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        kept = dropped = 0
        n_windows = max(0, (audio.shape[0] - WINDOW_SAMPLES) // HOP_WINDOW_SAMPLES + 1)
        for i in range(n_windows):
            start = i * HOP_WINDOW_SAMPLES
            chunk = audio[start : start + WINDOW_SAMPLES]
            if chunk.pow(2).mean().sqrt().item() < SILENCE_RMS:
                dropped += 1
                continue
            spec = extractor(chunk.unsqueeze(0))
            pt = out_dir / f"win_{i:04d}.pt"
            torch.save({"spectrogram": spec, "source": stem, "window_idx": i}, pt)
            if kept < a.png_per_file:
                save_png(spec, png_dir / f"{stem}_{i:04d}.png")
            rows.append({
                "path": pt.as_posix(),
                "label": "UNKNOWN",
                "speaker": stem,
                "source_file": src.as_posix(),
                "window_idx": i,
            })
            kept += 1

        total_win += kept
        total_dropped += dropped
        print(f"  {src.name[:46]:46s} -> {kept:4d} windows ({dropped} silent dropped)")

    with open(a.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "speaker",
                                          "source_file", "window_idx"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{total_win} windows kept, {total_dropped} silent dropped")
    print(f"  audio      -> {proc_dir}/")
    print(f"  tensors    -> {win_dir}/")
    print(f"  images     -> {png_dir}/")
    print(f"  manifest   -> {a.manifest}")
    print(f"\nEvery row is label=UNKNOWN. Fill in the label column "
          f"(human / ai) before training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
