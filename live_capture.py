"""
Bridge between the Android capture (android/CallAudioCapture.kt) and the
Python feature pipeline.

CallAudioCapture emits a FloatArray of 16000 mono samples in [-1, 1] every
0.5 s. This module turns those windows into model-ready [1, 257, 63] tensors
using features.py - the SAME extractor training uses, so there is no
train/serve skew on the Python side.

Three entry points:

  1. HTTP server - the phone POSTs each window to /ingest.
         python live_capture.py --serve --port 8765
     Wire format: raw little-endian float32 bytes (16000 floats = 64000 bytes),
     Content-Type: application/octet-stream. See android/WindowUploader.kt.

  2. Offline replay - feed a WAV through as if it were arriving live. Use this
     to test the whole path with no phone involved.
         python live_capture.py --wav my-folder/Demo_Podcast_1.mp3.wav

  3. Library - StreamingSpectrogram, for continuous byte streams.

SCOPE: this is the TRAINING-TIME and development bridge. It records and
featurises; it does not classify, because no model is trained yet. Once one
exists, replace `score_window()` below. For the actual demo the inference step
belongs on-device in Kotlin (TFLite) - Python is not running on the phone.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import wave
from pathlib import Path

import torch

from features import (
    FEATURE_SHAPE,
    HOP_WINDOW_SAMPLES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    LogLinearSpectrogram,
    spectrogram_from_window,
)

# How many recent window scores to average before reporting anything. Single
# windows are noisy and produce flickering false alarms; ~5-8 s of context is
# the useful unit. At a 0.5 s hop, 12 windows = 6 s.
SMOOTHING_WINDOWS = 12


# ------------------------------------------------------------------ decoding

def decode_float32_le(payload: bytes) -> torch.Tensor:
    """Raw little-endian float32 bytes -> [N] float tensor."""
    if len(payload) % 4 != 0:
        raise ValueError(
            f"payload of {len(payload)} bytes is not a whole number of float32s"
        )
    n = len(payload) // 4
    return torch.tensor(struct.unpack(f"<{n}f", payload), dtype=torch.float32)


def decode_pcm16_le(payload: bytes) -> torch.Tensor:
    """Raw little-endian int16 PCM -> [N] float tensor in [-1, 1]."""
    if len(payload) % 2 != 0:
        raise ValueError(f"payload of {len(payload)} bytes is not whole int16s")
    n = len(payload) // 2
    ints = struct.unpack(f"<{n}h", payload)
    return torch.tensor(ints, dtype=torch.float32) / 32768.0


# ----------------------------------------------------------------- streaming

class StreamingSpectrogram:
    """
    Continuous sample stream -> a spectrogram every `hop` samples.

    Use this when the source is a raw stream rather than pre-cut windows (the
    --wav replay, or a socket delivering arbitrary chunk sizes). If the phone
    is sending complete 1 s windows, call spectrogram_from_window() directly -
    re-windowing already-windowed data would double-count overlap.
    """

    def __init__(self, window: int = WINDOW_SAMPLES, hop: int = HOP_WINDOW_SAMPLES,
                 band_crop: bool = False):
        self.window = window
        self.hop = hop
        self.extractor = LogLinearSpectrogram(band_crop=band_crop)
        self._buf = torch.zeros(0, dtype=torch.float32)

    def push(self, samples) -> list[torch.Tensor]:
        """Add samples; return a (possibly empty) list of ready spectrograms."""
        if not isinstance(samples, torch.Tensor):
            samples = torch.as_tensor(samples, dtype=torch.float32)
        self._buf = torch.cat([self._buf, samples.flatten().to(torch.float32)])

        out: list[torch.Tensor] = []
        while self._buf.shape[0] >= self.window:
            chunk = self._buf[: self.window]
            out.append(self.extractor(chunk.unsqueeze(0)))
            self._buf = self._buf[self.hop :]
        return out

    def flush(self) -> list[torch.Tensor]:
        """Emit a final zero-padded window if a partial tail remains."""
        if self._buf.shape[0] == 0:
            return []
        tail = self._buf
        self._buf = torch.zeros(0, dtype=torch.float32)
        return [spectrogram_from_window(tail, self.extractor)]


class ScoreSmoother:
    """Rolling mean over recent window scores, per the 5-8 s averaging rule."""

    def __init__(self, n: int = SMOOTHING_WINDOWS):
        self.n = n
        self._scores: list[float] = []

    def add(self, score: float) -> float:
        self._scores.append(float(score))
        if len(self._scores) > self.n:
            self._scores.pop(0)
        return sum(self._scores) / len(self._scores)

    @property
    def ready(self) -> bool:
        """False until enough context has accumulated to report honestly."""
        return len(self._scores) >= self.n

    def reset(self) -> None:
        self._scores.clear()


# ----------------------------------------------------------------- inference

class Detector:
    """
    Loads a training/train.py checkpoint and scores windows.

    Reads band_crop back out of the checkpoint and builds its feature extractor
    to match, so a model trained on cropped features cannot silently be fed full
    ones - the mismatch that produces confident nonsense.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        from training.model import SpoofCNN

        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        self.band_crop = bool(ckpt.get("band_crop", False))
        self.n_freq_bins = int(ckpt.get("n_freq_bins", 257))
        self.holdout = ckpt.get("holdout_attacks") or []

        self.model = SpoofCNN(n_freq_bins=self.n_freq_bins).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.device = device
        self.extractor = LogLinearSpectrogram(band_crop=self.band_crop)

    def spectrogram(self, samples) -> torch.Tensor:
        return spectrogram_from_window(samples, self.extractor)

    def score(self, spec: torch.Tensor) -> float:
        """P(AI voice) for one [1, F, T] spectrogram."""
        with torch.no_grad():
            logit = self.model(spec.unsqueeze(0).to(self.device))
            return torch.sigmoid(logit).item()

    def describe(self) -> str:
        band = f"band-cropped [{self.n_freq_bins} bins]" if self.band_crop \
            else f"full band [{self.n_freq_bins} bins]"
        held = f", held out {self.holdout}" if self.holdout else ""
        return f"{band}{held}"


def score_window(spec: torch.Tensor, detector: "Detector | None" = None) -> float:
    """
    P(AI voice) for one window, or 0.5 when no model is loaded.

    Deliberately not faked when there are no weights - a made-up score would make
    the demo look like it works while measuring nothing.
    """
    return 0.5 if detector is None else detector.score(spec)


# -------------------------------------------------------------------- replay

def replay_wav(path: Path, verbose: bool = True, detector=None) -> int:
    """Push a WAV through the streaming path as if it arrived from the phone."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data).mean(dim=1)   # mono-ise
    if sr != SAMPLE_RATE:
        import torchaudio
        audio = torchaudio.functional.resample(audio.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)
        if verbose:
            print(f"resampled {sr} -> {SAMPLE_RATE} Hz")

    stream = StreamingSpectrogram(band_crop=getattr(detector, 'band_crop', False))
    smoother = ScoreSmoother()
    n = 0
    # 4096-sample chunks, i.e. deliberately not aligned to the window size, so
    # the ring buffer's partial-chunk handling actually gets exercised.
    for i in range(0, audio.shape[0], 4096):
        for spec in stream.push(audio[i : i + 4096]):
            n += 1
            avg = smoother.add(score_window(spec, detector))
            if verbose and n % 10 == 0:
                state = f"{avg:.3f}" if smoother.ready else "warming up"
                print(f"  window {n:4d}  shape={tuple(spec.shape)}  smoothed={state}")
    for spec in stream.flush():
        n += 1
        smoother.add(score_window(spec, detector))

    dur = audio.shape[0] / SAMPLE_RATE
    print(f"{path.name}: {dur:.1f}s -> {n} windows of {tuple(FEATURE_SHAPE)}")
    return n


# -------------------------------------------------------------------- server

def serve(host: str, port: int, save_dir: Path | None, detector=None) -> None:
    """Minimal HTTP endpoint the phone POSTs capture windows to."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"count": 0}
    smoother = ScoreSmoother()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass  # too chatty at two windows a second

        def _reply(self, code: int, body: str):
            payload = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self._reply(200, f'{{"ok":true,"windows":{state["count"]}}}')
            else:
                self._reply(404, '{"error":"try /health or POST /ingest"}')

        def do_POST(self):
            if self.path.split("?")[0] != "/ingest":
                self._reply(404, '{"error":"POST to /ingest"}')
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self._reply(400, '{"error":"bad Content-Length"}')
                return
            if length <= 0:
                self._reply(400, '{"error":"empty body"}')
                return

            payload = b""
            while len(payload) < length:
                part = self.rfile.read(length - len(payload))
                if not part:
                    break
                payload += part

            try:
                fmt = "pcm16" if "pcm16" in self.path else "float32"
                samples = (decode_pcm16_le(payload) if fmt == "pcm16"
                           else decode_float32_le(payload))
                spec = (detector.spectrogram(samples) if detector
                        else spectrogram_from_window(samples))
            except Exception as e:
                self._reply(400, f'{{"error":"{type(e).__name__}: {e}"}}')
                return

            state["count"] += 1
            avg = smoother.add(score_window(spec, detector))
            if save_dir:
                torch.save(
                    {"spectrogram": spec, "ts": time.time()},
                    save_dir / f"window_{state['count']:06d}.pt",
                )
            if state["count"] % 20 == 0:
                print(f"  {state['count']} windows  smoothed={avg:.3f}")
            self._reply(
                200,
                f'{{"n":{state["count"]},"score":{avg:.4f},'
                f'"ready":{str(smoother.ready).lower()}}}',
            )

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"listening on http://{host}:{port}")
    print(f"  model: {detector.describe() if detector else 'NONE - every window scores 0.5'}")
    print(f"  POST /ingest   raw float32-LE window ({WINDOW_SAMPLES} floats = "
          f"{WINDOW_SAMPLES * 4} bytes)")
    print(f"  POST /ingest?fmt=pcm16   raw int16-LE instead")
    print(f"  GET  /health")
    if save_dir:
        print(f"  saving windows to {save_dir.resolve()}")
    print("Phone and laptop must be on the same network; use the laptop's LAN IP.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        srv.server_close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="live_capture.py",
        description="Bridge Android capture windows into the Python feature pipeline",
    )
    p.add_argument("--serve", action="store_true", help="Run the HTTP ingest server")
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: all interfaces)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--save-dir", default=None, help="Persist each window as a .pt")
    p.add_argument("--wav", help="Offline: replay an audio file through the live path")
    p.add_argument("--model", help="Path to a training/train.py checkpoint (.pt). "
                                   "Without it every window scores 0.5.")
    a = p.parse_args(argv)

    detector = None
    if a.model:
        if not Path(a.model).is_file():
            print(f"checkpoint not found: {a.model}")
            return 1
        detector = Detector(a.model)
        print(f"loaded model: {detector.describe()}")

    if a.wav:
        path = Path(a.wav)
        if not path.is_file():
            print(f"file not found: {path}")
            return 1
        replay_wav(path, detector=detector)
        return 0
    if a.serve:
        serve(a.host, a.port, Path(a.save_dir) if a.save_dir else None, detector)
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
