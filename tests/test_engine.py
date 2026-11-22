import asyncio
import time

import pytest

from live_lyrics import engine as eng
from live_lyrics import recognize
from live_lyrics.transcribe import Word
from .conftest import noise


def match(key="k1", title="Song", artist="Artist", offset=30.0, **kw):
    return recognize.Match(key=key, title=title, artist=artist, album=None,
                           offset=offset, cover=None, **kw)


FAKE_LYRICS = {"synced": [{"t": 0, "text": "the quick brown fox jumps over the lazy dog"},
                          {"t": 5, "text": "a completely different second verse appears right here"},
                          {"t": 10, "text": "and the chorus rises high above the silent crowd"}],
               "duration": 180, "track_duration": 180}


@pytest.fixture
def no_lyrics_fetch(monkeypatch):
    async def fetch(title, artist, adamid=None, album=None):
        return dict(FAKE_LYRICS)
    monkeypatch.setattr(eng.lyrics, "fetch", fetch)


# ------------------------------------------------------------------ clip choice
async def wait_for_call(e, n=1, timeout=1.0):
    t0 = time.time()
    while len(e.identify_clips) < n and time.time() - t0 < timeout:
        await asyncio.sleep(0.01)
    return e.identify_clips


async def test_identify_clip_grows_with_time_since_onset(make_engine, source):
    source.push(noise(25))
    for ago, want in ((4.5, 4.5), (8, 8.0), (20, eng.CLIP)):
        e = make_engine()
        e.present, e.present_since, e.next_idle_check = True, time.time() - ago, 0.0
        e.maybe_shazam()
        clips = await wait_for_call(e)
        assert clips and abs(clips[-1] - want) < 0.3


async def test_suspect_checks_use_short_clip(make_engine, source):
    source.push(noise(25))
    e = make_engine()
    e.mode, e.force_check = "recorded", "words don't fit"
    e.maybe_shazam()
    clips = await wait_for_call(e)
    assert clips and abs(clips[-1] - eng.CLIP_SUSPECT) < 0.01


async def test_recheck_uses_full_window(make_engine, source):
    source.push(noise(25))
    e = make_engine()
    e.mode, e.misses = "recorded", 1
    e.maybe_shazam()
    clips = await wait_for_call(e)
    assert clips and abs(clips[-1] - eng.CLIP) < 0.01


async def test_no_lookup_when_blocked_or_busy(make_engine, source):
    source.push(noise(25))
    e = make_engine()
    e.mode, e.misses = "recorded", 1
    e.blocked_until = time.time() + 10
    e.maybe_shazam()
    await asyncio.sleep(0.05)
    assert not e.identify_clips
    e.blocked_until, e.busy = 0.0, True
    e.maybe_shazam()
    await asyncio.sleep(0.05)
    assert not e.identify_clips


# ------------------------------------------------------------------ budget
def test_budget_takes_and_refills():
    b = eng.Budget()
    start = b.available()
    b.take()
    assert b.available() == pytest.approx(start - 1, abs=0.01)
    b.drain()
    assert b.available() == pytest.approx(0, abs=0.01)
    b.stamp -= eng.BUCKET_REFILL  # pretend a refill interval passed
    assert b.available() == pytest.approx(1, abs=0.01)


# ------------------------------------------------------------------ match handling
async def test_on_match_locks_song_and_shows_lyrics(make_engine, no_lyrics_fetch):
    e = make_engine()
    await e.on_match(match(), clip_start=time.time() - 2, reason="identify")
    assert e.mode == "recorded" and e.state["status"] == "playing"
    assert e.state["title"] == "Song" and e.state["pos"] == 30.0
    assert e.song_text is not None and "chorus" in e.song_text.fulltext


async def test_same_match_reconfirms_without_relock(make_engine, no_lyrics_fetch):
    e = make_engine()
    await e.on_match(match(), clip_start=time.time() - 2, reason="identify")
    gen = e.generation
    await e.on_match(match(offset=40.0), clip_start=time.time(), reason="unconfirmed")
    assert e.generation == gen and e.state["pos"] == 40.0 and e.misses == 0


async def test_misses_drop_the_song_eventually(make_engine, no_lyrics_fetch):
    e = make_engine()
    await e.on_match(match(), clip_start=time.time() - 2, reason="identify")
    for _ in range(eng.RECORDED_GIVE_UP):
        await e.on_miss()
    assert e.mode is None and e.state["status"] == "listening"


# ------------------------------------------------------------------ whisper verification
def words_at(text: str, start: float = 6.0) -> list[Word]:
    return [Word(w, start + i * 0.4, start + i * 0.4 + 0.3) for i, w in enumerate(text.split())]


async def test_verify_confirms_matching_words(make_engine, no_lyrics_fetch):
    e = make_engine()
    await e.on_match(match(offset=6.0), clip_start=time.time(), reason="identify")
    await e.verify(words_at("completely different second verse appears right here"), time.time() - 10)
    assert e.confirms == 1 and e.mismatches == 0


async def test_verify_flags_foreign_words(make_engine, no_lyrics_fetch):
    e = make_engine()
    await e.on_match(match(offset=6.0), clip_start=time.time(), reason="identify")
    await e.verify(words_at("completely different second verse appears right here"), time.time() - 10)
    e.alarm_until = 0.0
    for _ in range(eng.MISMATCH_CHECK):
        await e.verify(words_at("wrong unrelated gibberish lyrics from another unknown tune"), time.time() - 10)
    assert e.force_check == "words don't fit"


# ------------------------------------------------------------------ play history
async def test_new_song_is_recorded_in_history(make_engine, no_lyrics_fetch):
    from live_lyrics import library as lib
    e = make_engine()
    await e.on_match(match(), clip_start=time.time() - 2, reason="identify")
    await e.on_match(match(offset=40.0), clip_start=time.time(), reason="unconfirmed")  # reconfirm: no new play
    plays = lib.read_history(e.library.history_path)
    assert len(plays) == 1 and plays[0]["title"] == "Song"


# ------------------------------------------------------------------ runner-up (alternate) matches
ALT_LYRICS = {"synced": [{"t": 0, "text": "moonlight shivers across frozen water tonight my dear"},
                         {"t": 5, "text": "silver branches bending under winter snowfall slowly"},
                         {"t": 10, "text": "candle flames remember every forgotten december"}],
              "duration": 200, "track_duration": 200}


@pytest.fixture
def alt_world(monkeypatch):
    """Primary song has FAKE_LYRICS, the runner-up 'Other Song' has ALT_LYRICS."""
    async def fetch(title, artist, adamid=None, album=None):
        return dict(ALT_LYRICS) if title == "Other Song" else dict(FAKE_LYRICS)
    monkeypatch.setattr(eng.lyrics, "fetch", fetch)

    async def about(key):
        return match(key=key, title="Other Song", artist="Other Band") if key == "alt1" else None
    monkeypatch.setattr(eng.recognize, "about", about)


async def test_wrong_version_swaps_to_runner_up(make_engine, alt_world):
    e = make_engine()
    await e.on_match(match(offset=6.0, alternates=("alt1",)), clip_start=time.time(), reason="identify")
    await e.verify(words_at("completely different second verse appears right here"), time.time() - 10)
    assert e.confirms == 1
    e.alarm_until = 0.0
    for _ in range(eng.MISMATCH_DROP):   # the room is singing the OTHER song's words
        await e.verify(words_at("silver branches bending under winter snowfall slowly"), time.time() - 10)
    assert e.state["title"] == "Other Song" and e.mode == "recorded"
    assert e.state["pos"] == 5.0   # the clock starts where the singer was located
    assert "december" in e.song_text.fulltext


async def test_drop_when_no_alternate_fits(make_engine, alt_world):
    e = make_engine()
    await e.on_match(match(offset=6.0), clip_start=time.time(), reason="identify")   # no alternates kept
    await e.verify(words_at("completely different second verse appears right here"), time.time() - 10)
    e.alarm_until = 0.0
    for _ in range(eng.MISMATCH_DROP):
        await e.verify(words_at("klaatu barada nikto gibberish untranslatable zork phrases here"), time.time() - 10)
    assert e.mode is None and e.state["status"] == "listening"


async def test_lyricless_song_borrows_runner_up_listing(make_engine, monkeypatch):
    async def fetch(title, artist, adamid=None, album=None):
        return dict(ALT_LYRICS) if title == "Other Song" else None   # nothing under Shazam's listing
    monkeypatch.setattr(eng.lyrics, "fetch", fetch)

    async def about(key):
        return match(key=key, title="Other Song", artist="Other Band")
    monkeypatch.setattr(eng.recognize, "about", about)
    e = make_engine()
    await e.on_match(match(alternates=("alt1",)), clip_start=time.time(), reason="identify")
    assert e.state["title"] == "Song"                       # still shown under the recognized name
    assert e.song_text and "december" in e.song_text.fulltext   # but with the runner-up's lyrics


# ------------------------------------------------------------------ presence logging
async def test_presence_logs_both_transitions(make_engine, source, caplog):
    import logging
    e = make_engine()
    e.noise, e.smooth = 0.001, 0.001
    with caplog.at_level(logging.INFO, logger="live-lyrics"):
        source.push(noise(2, level=0.1))            # loud room
        for _ in range(eng.PRESENT_ON):
            await e.monitor()
        assert e.present
        source.buf[:] = 0                           # dead silence
        for _ in range(eng.PRESENT_OFF):
            source.push(noise(0.25, level=0.0))
            await e.monitor()
        assert not e.present
    msgs = [r.message for r in caplog.records]
    assert any("music started" in m for m in msgs) and any("music gone" in m for m in msgs)


# ------------------------------------------------------------------ dead input (docking station)
async def test_dead_input_is_flagged_and_recovery_relearns_the_room(make_engine, source):
    e = make_engine()
    source.push(noise(3, level=0.0))          # a dock's "default input": pure digital silence
    await e.monitor()
    assert e.dead_since and not e.state.get("mic_dead")
    e.dead_since = time.time() - eng.MIC_DEAD_AFTER - 1   # ...for a while now
    await e.monitor()
    assert e.state.get("mic_dead") is True
    source.push(noise(3, level=0.1))          # a real microphone takes over
    await e.monitor()
    assert not e.state.get("mic_dead") and e.dead_since is None
    assert e.noise is not None and e.noise > 0.01         # the room was relearned at the new level
