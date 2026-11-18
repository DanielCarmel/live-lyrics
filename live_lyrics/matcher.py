"""Fuzzy text matching of transcribed singing against lyrics.

- identify(): which song (from a set of candidates) are these words from?
- locate():   where in the song is the singer right now?
"""
import difflib
import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

MIN_IDENTIFY_WORDS = 6
IDENTIFY_THRESHOLD = 80
LOCATE_THRESHOLD = 70
WEAK_FIT = 65   # heard text loosely resembles some part of the lyrics: not evidence of a new song


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))  # accents, Hebrew niqqud
    s = re.sub(r"[^\w\s']|_", " ", s).replace("'", "")
    return " ".join(s.split())


@dataclass
class SongText:
    lines: list[str]            # display lines
    times: list[float] | None   # line start times if synced
    words: list[str]            # normalized, flattened
    word_line: list[int]        # line index of each word
    word_frac: list[float]      # position of each word within its line (0..1)
    fulltext: str

    @classmethod
    def from_lyrics(cls, lyr: dict | None) -> "SongText | None":
        if not lyr or not ("synced" in lyr or "plain" in lyr):
            return None
        if "synced" in lyr:
            lines, times = [l["text"] for l in lyr["synced"]], [l["t"] for l in lyr["synced"]]
        else:
            lines, times = lyr["plain"], None
        words, wl, wf = [], [], []
        for i, line in enumerate(lines):
            ws = norm(line).split()
            for k, w in enumerate(ws):
                words.append(w)
                wl.append(i)
                wf.append((k + 0.5) / len(ws))
        return cls(lines, times, words, wl, wf, " ".join(words))


def identify(heard: str, candidates: dict[str, SongText]) -> tuple[str, float] | None:
    """Best matching candidate key for the heard text, if it's confident and unambiguous."""
    q = " ".join(norm(heard).split()[-20:])
    if len(q.split()) < MIN_IDENTIFY_WORDS or not candidates:
        return None
    scored = sorted(((fuzz.partial_ratio(q, s.fulltext), k) for k, s in candidates.items() if s.words),
                    reverse=True)
    if not scored:
        return None
    best, key = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0
    if best >= IDENTIFY_THRESHOLD and best - runner_up >= 5:
        return key, best
    return None


def score_text(heard: str, song: SongText) -> float:
    return fuzz.partial_ratio(" ".join(norm(heard).split()[-20:]), song.fulltext)


def locate(heard: list[str], song: SongText, prev_word: int | None) -> tuple[int, float] | None:
    """Align the last few heard words to the lyrics.
    Returns (index in song.words of the LAST heard word, score) or None."""
    hw = [w for w in (norm(x) for x in heard) if w][-10:]
    k = len(hw)
    if k < 4 or len(song.words) < 4:
        return None
    q = " ".join(hw)
    best = (-1.0, None)
    for length in range(max(2, k - 2), k + 3):  # whisper drops / adds words
        for j in range(0, len(song.words) - length + 1):
            end = j + length - 1
            s = fuzz.ratio(q, " ".join(song.words[j:j + length]))
            if prev_word is not None:
                if end < prev_word - 3:
                    s -= 15            # songs rarely go backwards...
                elif end > prev_word + 80:
                    s -= 10            # ...or jump far ahead
                s -= abs(end - prev_word) * 0.01  # repeated chorus: prefer the nearest one
            if s > best[0]:
                best = (s, end)
    score, end = best
    return (end, score) if end is not None and score >= LOCATE_THRESHOLD else None


def line_starts(heard: list[str], song: SongText, end: int) -> list[tuple[int, int]]:
    """After locate() put the last heard word at lyric word `end`: pair up heard words with
    lyric words and return (index into heard, line index) for each word that BEGINS a lyric
    line. Those are the moments a line's timestamp can be checked against real time."""
    hw = [norm(x) for x in heard]
    base = max(0, end - len(hw) - 2)
    lw = song.words[base:end + 1]
    pairs = []
    for b in difflib.SequenceMatcher(a=hw, b=lw, autojunk=False).get_matching_blocks():
        if b.size < 2:
            continue  # a lone matching word is too easily a coincidence
        for o in range(b.size - 1):  # (the block's last word has no following word backing it up)
            li = base + b.b + o
            if li == 0 or song.word_line[li - 1] != song.word_line[li]:
                pairs.append((b.a + o, song.word_line[li]))
    return pairs


_SCRIPTS = (("he", 0x0590, 0x05FF), ("ar", 0x0600, 0x06FF), ("ko", 0xAC00, 0xD7AF),
            ("ja", 0x3040, 0x30FF), ("el", 0x0370, 0x03FF), ("th", 0x0E00, 0x0E7F))


def script_language(text: str) -> str | None:
    """Whisper language code when the lyrics' script pins the language down (Hebrew, Arabic...).
    None for Latin and other scripts shared by many languages: let Whisper detect."""
    letters = [ord(c) for c in text if c.isalpha()]
    if not letters:
        return None
    for lang, lo, hi in _SCRIPTS:
        if sum(lo <= c <= hi for c in letters) > 0.5 * len(letters):
            return lang
    return None
