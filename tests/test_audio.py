import numpy as np

from live_lyrics import audio
from .conftest import FakeSource, noise


def test_ring_wraps_and_returns_newest():
    r = FakeSource()  # 30 s buffer
    for k in range(35):  # 35 s, one second at a time, each second filled with its own index
        r.push(np.full(audio.SAMPLE_RATE, float(k) / 100, dtype=np.float32))
    out, _ = r.latest(5)
    assert len(out) == 5 * audio.SAMPLE_RATE
    assert out[0] == 0.30 and out[-1] == 0.34  # the five newest seconds, in order


def test_ring_latest_caps_at_what_was_written():
    r = FakeSource()
    r.push(np.ones(audio.SAMPLE_RATE, dtype=np.float32))
    out, _ = r.latest(10)
    assert len(out) == audio.SAMPLE_RATE


def test_to_wav_header_and_length():
    wav = audio.to_wav(np.zeros(audio.SAMPLE_RATE, dtype=np.float32))
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert len(wav) == 44 + audio.SAMPLE_RATE * 2  # 16-bit mono


def test_rms():
    assert audio.rms(np.zeros(100, dtype=np.float32)) == 0.0
    assert abs(audio.rms(np.full(100, 0.5, dtype=np.float32)) - 0.5) < 1e-6
    assert audio.rms(np.zeros(0, dtype=np.float32)) == 0.0


def test_find_gap_spots_silence_between_songs():
    s = np.concatenate([noise(5), np.zeros(int(0.5 * audio.SAMPLE_RATE), np.float32), noise(4, seed=1)])
    end = audio.find_gap(s)
    assert end is not None and 5.3 <= end <= 5.8  # the gap ENDS where the next song starts


def test_find_gap_ignores_short_dips_and_silence_only():
    s = np.concatenate([noise(5), np.zeros(int(0.1 * audio.SAMPLE_RATE), np.float32), noise(4, seed=1)])
    assert audio.find_gap(s) is None            # a 0.1 s dip is staccato, not a track change
    assert audio.find_gap(np.zeros(10 * audio.SAMPLE_RATE, np.float32)) is None


def test_high_pass_removes_rumble_keeps_voice():
    t = np.arange(10 * audio.SAMPLE_RATE) / audio.SAMPLE_RATE
    rumble = 0.5 * np.sin(2 * np.pi * 50 * t)          # below the cutoff
    voice = 0.3 * np.sin(2 * np.pi * 440 * t)          # well above it
    out = audio.high_pass((rumble + voice).astype(np.float32))
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / audio.SAMPLE_RATE)
    assert spec[np.argmin(np.abs(freqs - 50))] < 0.01 * spec[np.argmin(np.abs(freqs - 440))]
    assert abs(audio.rms(out) - audio.rms(voice.astype(np.float32))) < 0.01  # the voice survived intact


def test_high_pass_passes_tiny_windows_through():
    x = np.zeros(100, np.float32)
    assert audio.high_pass(x) is x
