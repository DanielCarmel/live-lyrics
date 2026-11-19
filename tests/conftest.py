"""Shared pieces: a fake audio source and an engine wired to stubs instead of the network."""
import numpy as np
import pytest

from live_lyrics import audio
from live_lyrics.engine import Engine
from live_lyrics.library import Library


class FakeSource(audio._Ring):
    """A ring buffer that tests fill by hand."""

    def start(self):
        pass

    def stop(self):
        pass


def noise(seconds: float, level: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(0, level, int(seconds * audio.SAMPLE_RATE))).astype(np.float32)


@pytest.fixture
def source():
    return FakeSource()


@pytest.fixture
def make_engine(tmp_path, source):
    """Engine with no transcriber, an empty library in tmp, and a recorded stub for Shazam."""

    def build(identify=None, **kw):
        calls = []

        async def default_identify(wav: bytes):
            calls.append((len(wav) - 44) / 2 / audio.SAMPLE_RATE)  # seconds of PCM sent
            return None

        e = Engine(source, None, Library(str(tmp_path / "library.json")),
                   identify=identify or default_identify, **kw)
        e.identify_clips = calls   # seconds per call, when the default stub is used
        e.last_call = e.blocked_until = 0.0
        return e

    return build
