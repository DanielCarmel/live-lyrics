"""Song identification via Shazam (free, unofficial, no API key)."""
from dataclasses import dataclass

from aiohttp_retry import ExponentialRetry
from shazamio import Shazam
from shazamio.client import HTTPClient

# shazamio's default retries up to 20 times with backoff up to 60 s (incl. on 429 rate limits),
# which freezes recognition for ages. Fail fast instead; we simply try again with fresh audio.
_shazam = Shazam(http_client=HTTPClient(retry_options=ExponentialRetry(
    attempts=2, start_timeout=0.5, max_timeout=2, statuses={500, 502, 503, 504})))


@dataclass
class Match:
    key: str          # stable Shazam track id
    title: str
    artist: str
    album: str | None
    offset: float     # seconds into the track where the recorded clip STARTS
    cover: str | None
    adamid: str | None = None   # Apple Music id, used to look up the exact track length
    genre: str | None = None    # e.g. "Pop", "Hip-Hop/Rap": picks the display's text style


async def identify(audio: bytes | str) -> Match | None:
    out = await _shazam.recognize(audio)
    track, matches = out.get("track"), out.get("matches") or []
    if not track or not matches:
        return None
    album = None
    for section in track.get("sections", []):
        for m in section.get("metadata", []) or []:
            if m.get("title") == "Album":
                album = m.get("text")
    adamid = next((a.get("id") for a in (track.get("hub") or {}).get("actions", [])
                   if a.get("type") == "applemusicplay"), None)
    return Match(
        key=str(track.get("key")),
        title=track.get("title", ""),
        artist=track.get("subtitle", ""),
        album=album,
        offset=float(matches[0].get("offset", 0.0)),
        cover=(track.get("images") or {}).get("coverart"),
        adamid=adamid,
        genre=(track.get("genres") or {}).get("primary"),
    )
