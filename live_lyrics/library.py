"""Local lyrics library: songs from a setlist file + every song recognized so far.
Lets a live band's songs be identified purely from the sung words, with no API keys."""
import asyncio
import json
import logging
from pathlib import Path

from . import lyrics
from .matcher import SongText, norm

log = logging.getLogger("live-lyrics")


def song_key(title: str, artist: str) -> str:
    return f"{norm(artist)}|{norm(lyrics.clean_title(title))}"


class Library:
    def __init__(self, path: str = "data/library-v2.json"):
        self.path = Path(path)
        self.songs: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.songs = json.loads(self.path.read_text())
            except ValueError:   # cut short (e.g. the disk filled up mid-write): start afresh, songs are re-learned
                log.warning("song library %s is damaged; starting a new one", self.path)
        self.texts = {k: SongText.from_lyrics(s["lyrics"]) for k, s in self.songs.items()}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # write beside it, then swap: a failed write (full disk, power cut) can't wreck the library
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self.songs, ensure_ascii=False))
            tmp.replace(self.path)
        except OSError as e:
            log.warning("could not save the song library: %s", e)

    async def add(self, title: str, artist: str, lyr: dict | None = None) -> str | None:
        key = song_key(title, artist)
        if key in self.songs:
            return key
        lyr = lyr or await lyrics.fetch(title, artist)
        if not lyr:
            return None
        self.songs[key] = {"title": title, "artist": artist, "lyrics": lyr}
        self.texts[key] = SongText.from_lyrics(lyr)
        self.save()
        return key

    def set_offset(self, key: str, offset: float):
        """Remember a measured lyric timing correction for next time."""
        if key in self.songs and self.songs[key].get("offset") != offset:
            self.songs[key]["offset"] = offset
            self.save()

    def set_translation(self, key: str, lang: str, lines: list[str] | None):
        """Remember a song's translated lines (None: it's already in that language)."""
        if key in self.songs:
            self.songs[key].setdefault("translations", {})[lang] = lines
            self.save()

    async def load_setlist(self, path: str):
        """One song per line: `Artist - Title`. Lines starting with # are ignored."""
        entries = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and " - " in line:
                artist, title = (x.strip() for x in line.split(" - ", 1))
                entries.append((title, artist))
        keys = await asyncio.gather(*(self.add(t, a) for t, a in entries))
        for (t, a), k in zip(entries, keys):
            if not k:
                log.warning("setlist: no lyrics found for %s - %s", a, t)
        log.info("setlist: %d/%d songs have lyrics", sum(map(bool, keys)), len(entries))
