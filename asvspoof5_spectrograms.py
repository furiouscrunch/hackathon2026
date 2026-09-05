"""
ASVspoof5 protocol parser + PyTorch Dataset for the voice-scam-ai-detection project.

These .tsv files are PROTOCOL/METADATA files only — they tell you which audio
file goes with which label, they do not contain the audio itself. You still
need the actual .flac audio files from the ASVspoof5 release (via the
official request process at asvspoof.org) placed in a directory alongside
this metadata, matched by FILE_ID -> "<FILE_ID>.flac".

Column layout (space-separated, 10 columns), based on inspecting the files:
    1. SPEAKER_ID     e.g. T_4850 / D_0062 / E_1607
    2. FILE_ID        e.g. T_0000000000  -> maps to <FILE_ID>.flac
    3. GENDER         M / F
    4. CODEC          C01-C11 or '-' (only populated in eval_track_1)
    5. CODEC_Q        codec quality level (1-5) or '-'
    6. CODEC_SRC_REF  reference to source file used for codec application, or '-'
    7. ATTACK_CATEGORY  AC1/AC2/AC3 (attack category) or '-' for bonafide
    8. ATTACK_ID      A01-A26+ specific attack algorithm, or 'bonafide'
    9. LABEL          'bonafide' or 'spoof'  <-- the actual target label
    10. TRIM          reserved/unused, always '-' in what we've seen

Track 2 (speaker verification enroll/trial) files are a DIFFERENT task
(speaker verification, not spoof/AI-voice detection) and are intentionally
not covered by this loader — see the project notes on why.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from torch.utils.data import Dataset

from features import (
    FEATURE_SHAPE,
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    WIN_LENGTH,
    WINDOW_SECONDS,
    LogLinearSpectrogram,
)

try:
    import torchaudio
    from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
except ImportError as e:
    raise ImportError(
        "torchaudio is required. Install with: pip install torchaudio --break-system-packages"
    ) from e


LABEL_MAP = {"bonafide": 0, "spoof": 1}


@dataclass
class ASVspoofRecord:
    speaker_id: str
    file_id: str
    gender: str
    codec: Optional[str]
    codec_q: Optional[str]
    attack_category: Optional[str]
    attack_id: Optional[str]
    label: str  # 'bonafide' or 'spoof'

    @property
    def label_idx(self) -> int:
        return LABEL_MAP[self.label]


class LogMelSpectrogram:
    """
    Waveform [1, T] -> log-mel spectrogram [1, n_mels, frames].

    Defaults match common 16 kHz speech / anti-spoofing CNN pipelines:
    25 ms window, 10 ms hop, 80 mel bins.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: Optional[float] = 8000.0,
        top_db: float = 80.0,
    ):
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="htk",
        )
        self.to_db = AmplitudeToDB(stype="power", top_db=top_db)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # MelSpectrogram expects [..., time]
        spec = self.mel(waveform)
        spec = self.to_db(spec)
        return spec


def _decode_audio(audio_path: Path) -> tuple[torch.Tensor, int]:
    """
    Decode an audio file to (waveform [1, T] float32, sample_rate).

    torchaudio.load() routes through torchcodec on torchaudio 2.11+ (the default
    on Python 3.13/3.14), which needs FFmpeg's shared libraries at runtime and
    raises ImportError/RuntimeError when they are absent - the common case on a
    fresh macOS or Windows machine. Fall back to soundfile, which ships its own
    libsndfile and has no system-level dependency.
    """
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        return waveform, sr
    except Exception:
        try:
            import soundfile as sf
        except ImportError as e:
            raise RuntimeError(
                f"Could not decode {audio_path}: torchaudio.load() failed and "
                "soundfile is not installed. Install it with: pip install soundfile"
            ) from e
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        # soundfile gives [frames, channels]; torchaudio's convention is [channels, frames]
        waveform = torch.from_numpy(data).transpose(0, 1).contiguous()
        return waveform, sr


def parse_protocol_file(tsv_path: str | Path) -> list[ASVspoofRecord]:
    """Parse a train/dev/eval Track-1 protocol .tsv into a list of records."""
    records = []
    with open(tsv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if not row or len(row) < 9:
                continue
            speaker_id, file_id, gender = row[0], row[1], row[2]
            codec = row[3] if row[3] != "-" else None
            codec_q = row[4] if row[4] != "-" else None
            attack_category = row[6] if row[6] != "-" else None
            attack_id = row[7]
            label = row[8]
            if label not in LABEL_MAP:
                # Defensive: skip malformed / unexpected rows rather than crash
                continue
            records.append(
                ASVspoofRecord(
                    speaker_id=speaker_id,
                    file_id=file_id,
                    gender=gender,
                    codec=codec,
                    codec_q=codec_q,
                    attack_category=attack_category,
                    attack_id=None if attack_id == "bonafide" else attack_id,
                    label=label,
                )
            )
    return records


class ASVspoof5Dataset(Dataset):
    """
    PyTorch Dataset over ASVspoof5 Track-1 data (spoof/bonafide detection).

    Parameters
    ----------
    protocol_path : path to the .tsv protocol file (train / dev_track_1 / eval_track_1)
    audio_dir : directory containing the actual .flac audio files
    sample_rate : target sample rate to resample to (default 16000, standard for speech models)
    max_duration_s : if set, audio is center-cropped/padded to this many seconds.
                      Defaults to 1.0 s to match what CallAudioCapture emits on
                      the phone - train on the same window length you deploy on.
    feature : 'linear' (default) log linear-magnitude spectrogram [1, 257, 63],
              identical to the live capture path; 'mel' for the legacy log-mel
    transform : optional callable applied after spectrogram (or after waveform if
                return_type='waveform'). Use this for extra augmentation.
    audio_ext : file extension of the audio files (default '.flac', ASVspoof5's native format)
    skip_missing : if True (default), drop protocol rows whose audio file is absent
    return_type : 'spectrogram' (default) returns a log-mel spectrogram;
                  'waveform' returns the raw (resampled/cropped) waveform;
                  'both' includes waveform and spectrogram in the sample dict
    """

    def __init__(
        self,
        protocol_path: str | Path,
        audio_dir: str | Path,
        sample_rate: int = 16000,
        max_duration_s: Optional[float] = WINDOW_SECONDS,
        transform=None,
        audio_ext: str = ".flac",
        return_type: Literal["spectrogram", "waveform", "both"] = "spectrogram",
        feature: Literal["linear", "mel"] = "linear",
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: Optional[float] = 8000.0,
        skip_missing: bool = True,
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_s * sample_rate) if max_duration_s else None
        self.transform = transform
        self.audio_ext = audio_ext if audio_ext.startswith(".") else f".{audio_ext}"
        self.return_type = return_type
        self.skip_missing = skip_missing
        self._audio_index = self._build_audio_index()
        self.feature = feature
        if feature == "linear":
            # Default. Matches the live capture path exactly - see features.py.
            self.spectrogram_fn = LogLinearSpectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            )
        elif feature == "mel":
            # Opt-in only. Mel spacing smears the high band where vocoder
            # artifacts live, and the on-device path does not implement it.
            self.spectrogram_fn = LogMelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
            )
        else:
            raise ValueError(f"feature must be 'linear' or 'mel', got {feature!r}")

        records = parse_protocol_file(protocol_path)
        if skip_missing:
            kept = [r for r in records if r.file_id in self._audio_index]
            n_missing = len(records) - len(kept)
            if n_missing:
                print(
                    f"Skipping {n_missing} protocol rows with no matching "
                    f"*{self.audio_ext} under {self.audio_dir}"
                )
            records = kept
        self.records = records

        if len(self.records) == 0:
            raise ValueError(
                f"No usable records from {protocol_path}. "
                f"Need <FILE_ID>{self.audio_ext} files in {self.audio_dir} "
                "(protocol .tsv files are metadata only)."
            )

    def _build_audio_index(self) -> dict[str, Path]:
        """Map FILE_ID -> audio path, including files in subfolders."""
        index: dict[str, Path] = {}
        for path in self.audio_dir.rglob(f"*{self.audio_ext}"):
            index[path.stem] = path
        return index

    def __len__(self) -> int:
        return len(self.records)

    def _load_waveform(self, file_id: str) -> torch.Tensor:
        audio_path = self._audio_index.get(file_id)
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found for FILE_ID={file_id} under {self.audio_dir}. "
                "Make sure you've downloaded the actual ASVspoof5 audio "
                "(these .tsv files are metadata only) and pointed audio_dir at it."
            )
        waveform, sr = _decode_audio(audio_path)

        # Mono-ize (most ASVspoof audio is already mono, but be defensive)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Fix length: crop or pad so batches stack cleanly
        if self.max_samples is not None:
            n = waveform.shape[-1]
            if n > self.max_samples:
                start = (n - self.max_samples) // 2
                waveform = waveform[:, start : start + self.max_samples]
            elif n < self.max_samples:
                pad = self.max_samples - n
                waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform

    def __getitem__(self, idx: int):
        record = self.records[idx]
        waveform = self._load_waveform(record.file_id)

        sample = {
            "label": torch.tensor(record.label_idx, dtype=torch.long),
            "file_id": record.file_id,
            "attack_id": record.attack_id or "bonafide",
            "codec": record.codec or "none",
        }

        if self.return_type in ("spectrogram", "both"):
            spectrogram = self.spectrogram_fn(waveform)
            if self.transform is not None and self.return_type == "spectrogram":
                spectrogram = self.transform(spectrogram)
            sample["spectrogram"] = spectrogram

        if self.return_type in ("waveform", "both"):
            if self.transform is not None and self.return_type == "waveform":
                waveform = self.transform(waveform)
            sample["waveform"] = waveform

        return sample


def generate_spectrograms(
    protocol_path: str | Path,
    audio_dir: str | Path,
    out_dir: str | Path = "spectrograms",
    max_n: Optional[int] = 16,
    save_png: bool = True,
    save_pt: bool = False,
    **dataset_kwargs,
) -> list[Path]:
    """
    Build log-mel spectrograms for protocol rows that have matching audio.

    Writes PNG images (and optionally .pt tensors) named:
        <FILE_ID>_<label>_<attack_id>.png

    max_n : export this many items (None or a negative number = all matched files)
    """
    ds = ASVspoof5Dataset(
        protocol_path=protocol_path,
        audio_dir=audio_dir,
        return_type="spectrogram",
        skip_missing=True,
        **dataset_kwargs,
    )
    n = len(ds) if max_n is None or max_n < 0 else min(max_n, len(ds))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for i in range(n):
        sample = ds[i]
        stem = f"{sample['file_id']}_{sample['label'].item()}_{sample['attack_id']}"
        if save_png:
            png = out_dir / f"{stem}.png"
            save_spectrogram_png(sample["spectrogram"], png)
            written.append(png)
            print(f"wrote {png}  shape={tuple(sample['spectrogram'].shape)}")
        if save_pt:
            pt = out_dir / f"{stem}.pt"
            torch.save(
                {
                    "spectrogram": sample["spectrogram"].cpu(),
                    "label": sample["label"].cpu(),
                    "file_id": sample["file_id"],
                    "attack_id": sample["attack_id"],
                    "codec": sample["codec"],
                },
                pt,
            )
            written.append(pt)
            print(f"wrote {pt}")
    print(f"Generated {n} spectrogram(s) in {out_dir.resolve()}")
    return written


def save_spectrogram_png(spectrogram: torch.Tensor, out_path: str | Path) -> None:
    """Save a [1, n_mels, frames] (or [n_mels, frames]) tensor as a PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required to save PNG spectrograms. "
            "Install with: pip install matplotlib"
        ) from e

    spec = spectrogram.detach().cpu()
    if spec.ndim == 3:
        spec = spec.squeeze(0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.imshow(spec.numpy(), origin="lower", aspect="auto", cmap="magma")
    plt.xlabel("frames")
    plt.ylabel("mel bins")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def summarize_split(records: list[ASVspoofRecord], name: str) -> None:
    """Quick sanity-check printout of class/attack balance for a split."""
    n_bonafide = sum(1 for r in records if r.label == "bonafide")
    n_spoof = sum(1 for r in records if r.label == "spoof")
    attack_ids = sorted({r.attack_id for r in records if r.attack_id})
    codecs = sorted({r.codec for r in records if r.codec})
    print(f"[{name}] total={len(records)} bonafide={n_bonafide} spoof={n_spoof}")
    print(f"[{name}] attack IDs present: {attack_ids}")
    print(f"[{name}] codec conditions present: {codecs}")


def _running_in_notebook() -> bool:
    """True when executing inside Jupyter / Colab / IPython rather than a plain shell."""
    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        return True
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


def _strip_kernel_args(argv: list[str]) -> list[str]:
    """
    Remove the arguments Jupyter/Colab injects into sys.argv.

    A notebook kernel is launched as:
        colab_kernel_launcher.py -f /root/.../kernel-1234.json
    Left in place, '-f' makes argparse abort with SystemExit: 2, and the
    kernel .json path gets swallowed as the tsv_path positional.
    """
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "-f":
            skip_next = True
            continue
        if arg.startswith("-f=") or arg.startswith("--f="):
            continue
        if arg.endswith(".json") and "kernel" in Path(arg).name:
            continue
        cleaned.append(arg)
    return cleaned


NOTEBOOK_HELP = """\
No tsv_path given.

Running the CLI directly inside a notebook cell doesn't work, because the cell
inherits the kernel's own arguments. Do one of these instead:

  1) Run it as a real script from a shell cell:
       !python asvspoof5_spectrograms.py "/content/ASVspoof5.dev.track_1.tsv" \\
            --audio-dir /content/flac --out-dir /content/spectrograms --export-n 16

  2) Call main() with an explicit argument list:
       main(["/content/ASVspoof5.dev.track_1.tsv", "--audio-dir", "/content/flac"])

  3) Skip the CLI and use the functions:
       recs = parse_protocol_file(tsv)
       summarize_split(recs, name="dev.track_1")
       generate_spectrograms(protocol_path=tsv, audio_dir="/content/flac")
"""


def build_parser() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(
        prog="asvspoof5_spectrograms.py",
        description="Inspect ASVspoof5 protocol files and generate spectrograms",
    )
    parser.add_argument("tsv_path", help="Path to a protocol .tsv file")
    parser.add_argument(
        "--audio-dir",
        default=".",
        help="Directory of .flac files (searched recursively). Required to build spectrograms.",
    )
    parser.add_argument(
        "--out-dir",
        default="spectrograms",
        help="Where to write spectrogram PNGs / .pt files",
    )
    parser.add_argument(
        "--export-n",
        type=int,
        default=16,
        help="How many spectrograms to write. Use -1 for every file that has audio.",
    )
    parser.add_argument(
        "--save-pt",
        action="store_true",
        help="Also save each spectrogram tensor as a .pt file",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Do not write PNG images (use with --save-pt)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """
    Entry point. Safe to call from a notebook:

        main(["/content/ASVspoof5.dev.track_1.tsv", "--audio-dir", "/content/flac"])

    With argv=None the real command line is used, minus any kernel arguments.
    """
    in_notebook = _running_in_notebook()
    if argv is None:
        argv = _strip_kernel_args(sys.argv[1:])

    if not argv and in_notebook:
        print(NOTEBOOK_HELP)
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)

    tsv_path = Path(args.tsv_path)
    if not tsv_path.is_file():
        print(f"Protocol file not found: {tsv_path}")
        return 1

    recs = parse_protocol_file(tsv_path)
    summarize_split(recs, name=tsv_path.stem)
    if not recs:
        print(
            f"No valid rows parsed from {tsv_path}. Expected 10 space-separated "
            "columns per line (Track-1 protocol format)."
        )
        return 1

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        print(f"Audio directory not found: {audio_dir}")
        print("Protocol .tsv files are metadata only. Parser check finished.")
        return 0

    try:
        generate_spectrograms(
            protocol_path=tsv_path,
            audio_dir=audio_dir,
            out_dir=args.out_dir,
            max_n=args.export_n,
            save_png=not args.no_png,
            save_pt=args.save_pt,
        )
    except ValueError as e:
        print(e)
        print("Parser check finished. Add .flac files to generate spectrograms.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    # Don't raise SystemExit inside a notebook — it shows up as an ugly traceback.
    if not _running_in_notebook():
        sys.exit(exit_code)
