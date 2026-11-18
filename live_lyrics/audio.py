"""Continuous audio capture into a ring buffer (mic, or a file simulated in real time)."""
import io
import subprocess
import threading
import time
import wave

import numpy as np

SAMPLE_RATE = 16000  # both Shazam and Whisper want 16 kHz mono


class _Ring:
    def __init__(self, seconds: float = 30):
        self.buf = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
        self.written = 0          # total samples ever written
        self.end_wall = 0.0       # wall-clock time of the newest sample
        self.lock = threading.Lock()

    def push(self, x: np.ndarray):
        with self.lock:
            n = len(x)
            if n >= len(self.buf):
                self.buf[:] = x[-len(self.buf):]
            else:
                i = self.written % len(self.buf)
                first = min(n, len(self.buf) - i)
                self.buf[i:i + first] = x[:first]
                self.buf[:n - first] = x[first:]
            self.written += n
            self.end_wall = time.time()

    def latest(self, seconds: float) -> tuple[np.ndarray, float]:
        """Most recent `seconds` of audio (float32) and the wall time of its last sample."""
        with self.lock:
            n = min(int(seconds * SAMPLE_RATE), self.written, len(self.buf))
            i = self.written % len(self.buf)
            out = np.concatenate([self.buf[i:], self.buf[:i]])[-n:] if n else np.zeros(0, np.float32)
            return out.copy(), self.end_wall

    def available(self) -> float:
        return min(self.written, len(self.buf)) / SAMPLE_RATE


class Mic(_Ring):
    def __init__(self, device=None):
        super().__init__()
        import sounddevice as sd
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                     device=device, blocksize=1600,
                                     callback=lambda d, *_: self.push(d[:, 0]))

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()


class FileSource(_Ring):
    """Plays an audio file into the buffer in real time, as if it came from the mic.
    `af` is an optional ffmpeg audio filter, e.g. "atempo=1.06,asetrate=16000*1.03,aresample=16000"
    to make a recording sound like a different (live) performance."""

    def __init__(self, path: str, af: str | None = None):
        super().__init__()
        self.cmd = ["ffmpeg", "-loglevel", "error", "-re", "-i", path]
        if af:
            self.cmd += ["-af", af]
        self.cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]

    def start(self):
        def run():
            p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)
            while chunk := p.stdout.read(SAMPLE_RATE // 10 * 4):
                self.push(np.frombuffer(chunk, dtype=np.float32))
        threading.Thread(target=run, daemon=True).start()


def to_wav(samples: np.ndarray) -> bytes:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0


def find_gap(samples: np.ndarray, noise: float | None = None, frame: float = 0.1,
             min_len: float = 0.3) -> float | None:
    """Seconds into `samples` where the most recent gap ENDS (e.g. the blip of silence when a
    player skips tracks). A gap is >= `min_len` s that sounds like the empty room - near the
    noise floor if we know it, else far below the window's typical loudness - with music on
    both sides. Short dips inside the music (staccato, drum breaks) don't count."""
    n = int(frame * SAMPLE_RATE)
    if len(samples) < n * 20:
        return None
    frames = samples[: len(samples) // n * n].reshape(-1, n)
    levels = np.sqrt((frames ** 2).mean(axis=1))
    typical = float(np.median(levels))
    if typical < 0.0003:
        return None
    limit = typical * 0.15
    if noise and typical > 1.6 * noise:
        limit = max(limit, min(noise * 1.35, typical * 0.6))
    quiet = levels < limit
    need = round(min_len / frame)
    run, end = 0, None
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run >= need and i + 1 < len(quiet) and not quiet[i + 1]:
            end = i + 1
    return end * frame if end is not None else None
