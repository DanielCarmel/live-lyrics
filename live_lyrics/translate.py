"""Optional lyric translation with Google's free web endpoint (unofficial, no API key).

A song is translated once, as a whole, when it is identified; the result is kept in the library.
Lines stay one-to-one with the original, so all timing still applies."""
import asyncio
import logging

import httpx

log = logging.getLogger("live-lyrics")
LANGUAGES = {"en": "English", "he": "Hebrew", "es": "Spanish"}
URL = "https://clients5.google.com/translate_a/t"   # takes many lines per request, answers line for line
CHUNK = 1500   # characters per request
last_failed = False   # the last translate_lines() gave None because of an error (not: same language)


async def _chunk(client: httpx.AsyncClient, lines: list[str], target: str) -> tuple[list[str] | None, str | None]:
    """Translate lines in one request. (translated lines, or None if they didn't come back one-to-one; source language)."""
    r = await client.post(URL, params={"client": "dict-chrome-ex", "sl": "auto", "tl": target},
                          data={"q": lines})
    r.raise_for_status()
    data = r.json()
    if len(data) != len(lines):
        return None, None
    out = [(d[0] if isinstance(d, list) else d).strip() for d in data]
    langs = [d[1] for d in data if isinstance(d, list) and len(d) > 1]
    return (out if all(out) else None), (max(set(langs), key=langs.count) if langs else None)


async def translate_lines(lines: list[str], target: str) -> list[str] | None:
    """The same lines in `target` ("en"/"he"/"es"); empty lines stay empty.
    None if the song is already in that language or the translation failed."""
    global last_failed
    last_failed = False
    idx = [i for i, l in enumerate(lines) if l and l.strip()]
    if not idx:
        return None
    chunks, cur, size = [], [], 0
    for i in idx:
        if cur and size + len(lines[i]) > CHUNK:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += len(lines[i]) + 1
    chunks.append(cur)
    out = list(lines)
    try:
        async with httpx.AsyncClient(timeout=8, headers={"User-Agent": "Mozilla/5.0"}) as c:
            results = await asyncio.gather(*(_chunk(c, [lines[i] for i in ch], target) for ch in chunks))
            langs = [lang for _, lang in results if lang]
            if langs and max(set(langs), key=langs.count) == target:
                return None  # already in that language
            for ch, (tr, _) in zip(chunks, results):
                if tr is None:  # lines got merged or split: ask for this part line by line
                    singles = await asyncio.gather(*(_chunk(c, [lines[i]], target) for i in ch))
                    tr = [s[0][0] if s[0] else lines[i] for s, i in zip(singles, ch)]
                for i, t in zip(ch, tr):
                    out[i] = t
    except Exception as e:
        log.warning("translation failed: %s", e)
        last_failed = True
        return None
    return out


def lines_of(lyr: dict) -> list[str]:
    return [l["text"] for l in lyr["synced"]] if "synced" in lyr else list(lyr["plain"])


def with_lines(lyr: dict, lines: list[str]) -> dict:
    """A copy of the lyrics with the line texts replaced (timing untouched)."""
    if "synced" in lyr:
        return {**lyr, "synced": [{**l, "text": t} for l, t in zip(lyr["synced"], lines)]}
    return {**lyr, "plain": lines}
