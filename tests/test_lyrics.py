from live_lyrics import lyrics


def test_parse_lrc_sorts_and_handles_multiple_stamps():
    lrc = "[00:10.5]late line\n[00:02]early\n[00:20][00:40]twice"
    lines = lyrics.parse_lrc(lrc)
    assert [l["t"] for l in lines] == [2, 10.5, 20, 40]
    assert lines[2]["text"] == lines[3]["text"] == "twice"


def test_clean_title():
    assert lyrics.clean_title("Song (feat. X) - Remastered 2011") == "Song"
    assert lyrics.clean_title("Plain Title") == "Plain Title"


def test_title_variants_dedupes_and_keeps_inner():
    v = lyrics.title_variants("Transliterated (מקור)")
    assert "Transliterated" in v and "מקור" in v
    assert len(v) == len(set(v))
    assert lyrics.title_variants("Song (Live)") == ["Song (Live)", "Song"]


def test_score_gates_on_title_and_artist_or_duration():
    titles = [lyrics.norm("My Song")]
    hit = {"trackName": "My Song", "artistName": "The Band", "duration": 200, "syncedLyrics": "x"}
    assert lyrics._score(hit, titles, "the band", 200) is not None
    assert lyrics._score({**hit, "trackName": "Other Thing"}, titles, "the band", 200) is None
    # artist in another script: the matching duration vouches for it
    assert lyrics._score({**hit, "artistName": "Другой"}, titles, "the band", 200.5) is not None
    # wrong artist AND wrong length: rejected
    assert lyrics._score({**hit, "artistName": "Другой", "duration": 150}, titles, "the band", 200) is None


# ------------------------------------------------------------------ fetch: exact fast path
import json

import httpx
import pytest


def transport(routes: dict, counter: dict):
    """routes: path -> (status, json). Counts hits per path."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        counter[path] = counter.get(path, 0) + 1
        for p, (status, body) in routes.items():
            if path.endswith(p):
                return httpx.Response(status, json=body)
        return httpx.Response(404, json={})
    return httpx.MockTransport(handler)


EXACT = {"trackName": "My Song", "artistName": "The Band", "duration": 200,
         "syncedLyrics": "[00:01]first line\n[00:05]second line", "plainLyrics": "first line\nsecond line"}
ITUNES = {"results": [{"trackTimeMillis": 200000}]}


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(lyrics, "_cache", {})


async def test_fetch_uses_exact_lookup_and_skips_search(monkeypatch):
    hits = {}
    monkeypatch.setattr(lyrics, "_transport", transport(
        {"/lookup": (200, ITUNES), "/api/get": (200, EXACT)}, hits))
    out = await lyrics.fetch("My Song", "The Band", adamid="123", album="The Album")
    assert out and out["synced"][0]["text"] == "first line" and out["track_duration"] == 200
    assert hits.get("/api/get") == 1 and "/api/search" not in hits


async def test_fetch_falls_back_to_search_when_exact_misses(monkeypatch):
    hits = {}
    monkeypatch.setattr(lyrics, "_transport", transport(
        {"/lookup": (200, ITUNES), "/api/get": (404, {}),
         "/api/search": (200, [{"id": 1, **EXACT}])}, hits))
    out = await lyrics.fetch("My Song", "The Band", adamid="123")
    assert out and "synced" in out
    assert hits.get("/api/get") == 1 and hits.get("/api/search", 0) >= 1


async def test_fetch_without_duration_goes_straight_to_search(monkeypatch):
    hits = {}
    monkeypatch.setattr(lyrics, "_transport", transport(
        {"/api/search": (200, [{"id": 1, **EXACT}])}, hits))
    out = await lyrics.fetch("My Song", "The Band")
    assert out and "synced" in out
    assert "/api/get" not in hits
