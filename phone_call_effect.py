#!/usr/bin/env python3
"""
phone_call_effect.py

Converts a clean audio file into "phone call quality" audio.

Two outputs are generated:
  1. <name>_bandlimited.wav  - bandpass filtered (300-3400Hz) + compressed,
                               resampled to 8kHz. Clean telephone bandwidth.
  2. <name>_mulaw.wav        - same as above, plus encoded through the real
                               G.711 mu-law codec (used by actual landline
                               and VoIP calls) for authentic codec character.

Requires ffmpeg to be installed and available on PATH.
  macOS:   brew install ffmpeg
  Windows: https://ffmpeg.org/download.html (add to PATH)
  Linux:   sudo apt install ffmpeg

Usage:
    python phone_call_effect.py input_audio.wav
    python phone_call_effect.py input_audio.wav -o output_folder
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Filter chain applied before resampling, so there is nothing left above
# the new Nyquist frequency to alias/distort when we drop to 8kHz.
FILTER_CHAIN = (
    "highpass=f=300,"
    "lowpass=f=3400,"
    "acompressor=threshold=-18dB:ratio=3:attack=5:release=50,"
    "aresample=8000"
)


def run_ffmpeg(args):
    result = subprocess.run(
        ["ffmpeg", "-y", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("ffmpeg failed - see error output above")


def make_bandlimited(input_path: Path, output_path: Path):
    """Clean phone-bandwidth version: filtered + compressed + resampled."""
    run_ffmpeg([
        "-i", str(input_path),
        "-af", FILTER_CHAIN,
        "-c:a", "pcm_s16le",
        str(output_path),
    ])


def make_mulaw(input_path: Path, encoded_path: Path, final_path: Path):
    """Authentic version: same processing, passed through real G.711 mu-law."""
    # Encode with the same filter chain, output as mu-law
    run_ffmpeg([
        "-i", str(input_path),
        "-af", FILTER_CHAIN,
        "-ar", "8000",
        "-c:a", "pcm_mulaw",
        str(encoded_path),
    ])
    # Decode back to standard PCM so it plays everywhere (artifacts stay baked in)
    run_ffmpeg([
        "-i", str(encoded_path),
        "-c:a", "pcm_s16le",
        str(final_path),
    ])


def process_file(input_path: Path, outdir: Path):
    stem = input_path.stem
    bandlimited_path = outdir / f"{stem}_bandlimited.wav"
    mulaw_encoded_path = outdir / f"{stem}_mulaw_encoded.wav"
    mulaw_final_path = outdir / f"{stem}_mulaw.wav"

    print(f"\nProcessing: {input_path.name}")
    print("  -> Creating bandlimited version...")
    make_bandlimited(input_path, bandlimited_path)
    print(f"     Saved: {bandlimited_path}")

    print("  -> Creating mu-law (authentic codec) version...")
    make_mulaw(input_path, mulaw_encoded_path, mulaw_final_path)
    if mulaw_encoded_path.exists():
        mulaw_encoded_path.unlink()  # intermediate file, not needed
    print(f"     Saved: {mulaw_final_path}")


def main():
    parser = argparse.ArgumentParser(description="Make audio sound like a phone call.")
    parser.add_argument("input", help="Path to input audio file or folder containing audio files")
    parser.add_argument("-o", "--outdir", default="phone_audio_out", help="Output directory (default: phone_audio_out)")
    parser.add_argument("--recursive", action="store_true", help="Recursively process subdirectories")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input path not found: {input_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    supported_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

    if input_path.is_dir():
        pattern = "**/*" if args.recursive else "*"
        audio_files = [
            f for f in input_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_exts and not f.name.endswith(("_bandlimited.wav", "_mulaw.wav"))
        ]
        if not audio_files:
            sys.exit(f"No audio files found in directory: {input_path}")
        print(f"Found {len(audio_files)} audio file(s) in {input_path}. Output folder: {outdir}")
        for audio_file in sorted(audio_files):
            process_file(audio_file, outdir)
    else:
        process_file(input_path, outdir)

    print("\nDone! The mu-law version is the closest to a real phone call.")


if __name__ == "__main__":
    main()
