"""Time-synced lyrics from LRCLIB (free, open, no API key).

LRCLIB often has several versions of a song (album / single / live / remaster) whose timestamps
differ, and titles may be stored in another script than Shazam reports. So we gather candidates
from several queries and pick the one that best matches title, artist and - most importantly -
the real track duration (looked up for free from the Apple Music id Shazam gives us)."""
import asyncio
import logging
import re

from urllib.parse import quote

import httpx
from rapidfuzz import fuzz

from .matcher import norm

log = logging.getLogger("live-lyrics")
API = "https://lrclib.net/api"
HEADERS = {"User-Agent": "live-lyrics-poc/0.1"}
_TS = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
_cache: dict[tuple[str, str], dict | None] = {}
_transport: httpx.AsyncBaseTransport | None = None   # tests inject a mock transport here


def parse_lrc(lrc: str) -> list[dict]:
    lines = []
    for raw in lrc.splitlines():
        stamps = _TS.findall(raw)
        text = _TS.sub("", raw).strip()
        for mm, ss in stamps:
            lines.append({"t": int(mm) * 60 + float(ss), "text": text})
    return sorted(lines, key=lambda l: l["t"])


def clean_title(title: str) -> str:
    # "Song (feat. X) - Remastered 2011" -> "Song"
    title = re.sub(r"\s*[\(\[](feat\.?|ft\.?|with)[^\)\]]*[\)\]]", "", title, flags=re.I)
    return re.split(r"\s+-\s+", title)[0].strip()


def title_variants(title: str) -> list[str]:
    """Different ways the same title may be stored: cleaned, without (...), and the (...) parts
    themselves - Shazam often reports `Transliterated (מקור)`."""
    inner = re.findall(r"[\(\[]([^\)\]]+)[\)\]]", title)
    bare = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]", "", title).strip()
    out = [title, clean_title(title), bare, clean_title(bare)]
    out += [p.strip() for p in inner if not re.match(r"(feat|ft|with|remaster|live|version|edit)\b", p.strip(), re.I)]
    return [v for v in dict.fromkeys(out) if v]


async def track_duration(client: httpx.AsyncClient, adamid: str | None) -> float | None:
    """Exact track length from Apple's free lookup API."""
    if not adamid:
        return None
    try:
        r = await client.get("https://itunes.apple.com/lookup", params={"id": adamid}, timeout=4)
        ms = r.json()["results"][0].get("trackTimeMillis")
        return ms / 1000 if ms else None
    except Exception:
        return None


async def _search(client: httpx.AsyncClient, **params) -> list[dict]:
    try:
        r = await client.get(f"{API}/search", params=params)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


async def _get_exact(client: httpx.AsyncClient, title: str, artist: str, album: str | None,
                     duration: float) -> dict | None:
    """One-request lookup by the track's exact signature (LRCLIB matches duration within 2 s).
    Most recognized songs answer here at once, skipping the whole search fan-out."""
    params = {"track_name": title, "artist_name": artist, "duration": round(duration)}
    if album:
        params["album_name"] = album
    try:
        r = await client.get(f"{API}/get", params=params)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _score(hit: dict, titles: list[str], artist: str, duration: float | None) -> float | None:
    ht, ha = norm(hit.get("trackName") or ""), norm(hit.get("artistName") or "")
    ht_bare = norm(re.sub(r"[\(\[][^\)\]]*[\)\]]", "", hit.get("trackName") or ""))
    title_sim = max(max(fuzz.ratio(ht, t), fuzz.ratio(ht_bare, t)) for t in titles)
    artist_sim = max(fuzz.token_set_ratio(ha, artist), fuzz.partial_ratio(ha, artist)) if ha and artist else 0
    off = abs((hit.get("duration") or 0) - duration) if duration else None
    same_length = off is not None and off <= 2.5
    # Must look like the same song: similar title AND (same artist OR same length).
    # (Artist names may be in another script, where only the length can vouch for it.)
    if title_sim < 80 or not (artist_sim >= 70 or same_length):
        return None
    score = title_sim + 0.3 * artist_sim + (15 if hit.get("syncedLyrics") else 0)
    if off is not None:
        score += 40 if same_length else -min(off, 30)
    return score


# ---------------------------------------------------------------- fallback sources
BROWSER = {"User-Agent": "Mozilla/5.0"}
_CREDIT = re.compile(r"^\s*(作词|作曲|编曲|制作人?|混音|母带|录音|吉他|贝斯|和声|监制|出品|发行|词|曲|"
                     r"lyrics by|written by|composed by|produced by|arranged by|lyricist|composer)\s*[:：]", re.I)


async def _netease(client: httpx.AsyncClient, variants: list[str], titles: list[str],
                   artist: str, duration: float | None) -> dict | None:
    """Synced lyrics from NetEase Cloud Music's public endpoints (large international catalogue)."""
    try:
        found = await asyncio.gather(*(
            client.get("https://music.163.com/api/search/get", headers=BROWSER,
                       params={"s": f"{v} {artist}", "type": 1, "limit": 10}) for v in variants[:2]))
        songs = {s["id"]: s for r in found for s in (r.json().get("result") or {}).get("songs", [])}
    except Exception:
        return None
    scored = []
    for s in songs.values():
        hit = {"trackName": s.get("name"), "artistName": ", ".join(a["name"] for a in s.get("artists", [])),
               "duration": (s.get("duration") or 0) / 1000, "syncedLyrics": True}
        sc = _score(hit, titles, norm(artist), duration)
        if sc is not None:
            scored.append((sc, s["id"], hit))
    for _, sid, hit in sorted(scored, key=lambda x: -x[0])[:2]:
        try:
            r = await client.get("https://music.163.com/api/song/lyric", headers=BROWSER,
                                 params={"id": sid, "lv": 1, "tv": -1})
            lines = [l for l in parse_lrc((r.json().get("lrc") or {}).get("lyric") or "")
                     if not _CREDIT.match(l["text"])]
        except Exception:
            continue
        if sum(1 for l in lines if l["text"]) >= 6:
            return {"synced": lines, "duration": hit["duration"], "source": "NetEase",
                    "found_as": f'{hit["artistName"]} - {hit["trackName"]}'}
    return None


async def _lyrics_ovh(client: httpx.AsyncClient, title: str, artist: str) -> dict | None:
    """Plain (unsynced) lyrics from lyrics.ovh."""
    try:
        r = await client.get(f"https://api.lyrics.ovh/v1/{quote(artist, safe='')}/{quote(clean_title(title), safe='')}", timeout=6)
        text = r.json().get("lyrics") or ""
    except Exception:
        return None
    lines = [l.strip() for l in text.replace("\r", "").split("\n")]
    if lines and lines[0].lower().startswith("paroles de la chanson"):
        lines = lines[1:]
    lines = [l for i, l in enumerate(lines) if l or (i and lines[i - 1])]  # collapse blank runs
    if sum(1 for l in lines if l) < 6:
        return None
    return {"plain": lines, "source": "lyrics.ovh", "found_as": f"{artist} - {title}"}


async def fetch(title: str, artist: str, adamid: str | None = None, album: str | None = None) -> dict | None:
    """{"synced": [{t, text}...] | "plain": [str...], "duration": lyrics' own length,
    "track_duration": real track length if known}. Without lyrics: just {"track_duration"} or None.

    Sources, best first: LRCLIB synced -> NetEase synced -> LRCLIB plain -> lyrics.ovh plain."""
    key = (title.lower(), artist.lower())
    if key in _cache:
        return _cache[key]
    variants = title_variants(title)
    titles = [norm(v) for v in variants]
    async with httpx.AsyncClient(headers=HEADERS, timeout=8, transport=_transport) as c:
        duration = await track_duration(c, adamid)
        best, best_score, seen = None, None, set()
        if duration:   # fast path: the exact signature, one request
            hit = await _get_exact(c, clean_title(title), artist, album, duration)
            if hit and hit.get("syncedLyrics") and _score(hit, titles, norm(artist), duration) is not None:
                best = hit
        if best is None:
            queries = [_search(c, track_name=v, artist_name=artist) for v in variants[:3]]
            queries += [_search(c, q=v) for v in variants if v not in (title,)][:3]
            results = await asyncio.gather(*queries)
            for hit in (h for r in results for h in r):
                if hit.get("id") in seen or not (hit.get("syncedLyrics") or hit.get("plainLyrics")):
                    continue
                seen.add(hit.get("id"))
                sc = _score(hit, titles, norm(artist), duration)
                if sc is not None and (best_score is None or sc > best_score):
                    best, best_score = hit, sc
        result = None
        if best and best.get("syncedLyrics"):
            result = {"synced": parse_lrc(best["syncedLyrics"])}
        else:
            # Fallbacks, asked together so a slow one doesn't hold up the screen
            async def capped(coro):
                try:
                    return await asyncio.wait_for(coro, 4.5)
                except Exception:
                    return None
            netease, ovh = await asyncio.gather(
                capped(_netease(c, variants, titles, artist, duration)),
                capped(_lyrics_ovh(c, title, artist)) if not best else asyncio.sleep(0))
            result = netease or ({"plain": best["plainLyrics"].splitlines()} if best else None) or ovh
        if result and "source" not in result:
            result.update(duration=best.get("duration"), source="LRCLIB",
                          found_as=f'{best.get("artistName")} - {best.get("trackName")}')
    if result:
        result["track_duration"] = duration
        log.info("lyrics: %s, %s (%s)%s", result["source"], "synced" if "synced" in result else "plain",
                 result.pop("found_as"), f", track is {duration:.0f}s" if duration else "")
    else:
        log.info("lyrics: none found for %s - %s", artist, title)
        if duration:
            result = {"track_duration": duration}  # still lets us know when the song ends
    _cache[key] = result
    return result
