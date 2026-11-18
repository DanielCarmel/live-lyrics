"""The brain: decides when music is playing, what it is, where in the song we are, and when it ended.

Two recognizers work together:
  * Shazam   - identifies recordings and gives the exact position. Precise, but the free endpoint
               allows only a few lookups per minute, so every lookup has to earn its place.
  * Whisper  - local and unlimited. Confirms the words being sung still fit the current lyrics
               (so Shazam needn't be asked again), spots song changes, fine-tunes lyric timing,
               and follows live bands that Shazam can't recognize at all.
A fast loudness monitor decides whether there is music at all, so nothing is sent anywhere
(and nothing is transcribed) while the room is just a room.
"""
import asyncio
import json
import logging
import time
from collections import deque

import numpy as np
from shazamio.exceptions import RateLimited

from . import audio, genius, lyrics, matcher, recognize, translate
from .library import Library, song_key
from .matcher import SongText

log = logging.getLogger("live-lyrics")

STEP = 2.0               # seconds between Whisper passes
CLIP = 6.0               # seconds of audio sent to Shazam
WINDOW = 10.0            # seconds of audio given to Whisper
AFTER_GAP = 4.0          # seconds of new audio needed before a Shazam lookup is worth it

# --- is there music? (loudness monitor, 4x per second) ---
PRESENT_RATIO = 1.5      # "sound" = this much louder than the room's noise floor...
PRESENT_ON = 6           # ...for this many consecutive checks (1.5 s) -> music started
PRESENT_OFF = 8          # this many consecutive quiet checks (2 s) -> music gone
STOP_AFTER = 3.0         # s at the empty room's level before the song counts as paused
STOP_AFTER_SOFT = 8.0    # s merely quieter than the song: a soft passage is far more likely than a pause
END_GRACE = 1.5          # seconds past the track's length -> finished

# --- Shazam budget. The free endpoint's limit is unpredictable: sometimes ~1 per 20 s is fine
# for minutes, sometimes 4 lookups in 30 s gets blocked. So: small bursts, slow refill. ---
BUCKET_SIZE = 4
BUCKET_REFILL = 25.0     # seconds per token
RESERVE = 2              # tokens kept back for real events (new song, song change)
MIN_SPACING = 6.0        # never two lookups closer than this
BLOCK_PAUSE = 30.0       # after Shazam says "rate limited"
TIMEOUT = 6.0
IDLE_BACKOFF = (8.0, 15.0, 30.0, 60.0)  # unidentified sound: retry ever more slowly
LONGSHOT_EVERY = (60.0, 120.0, 300.0)  # no clear music detected: still take a rare look (very quiet music)
# Locked song that nothing has confirmed for a while -> ask Shazam. How long "a while" is depends
# on how much budget is going spare (unused tokens don't accumulate, so a full bucket is free to use).
UNCONFIRMED_FULL = 15.0   # bucket full
UNCONFIRMED = 30.0        # otherwise (needs the reserve intact)
RETRY_AFTER_MISS = 8.0
RECORDED_GIVE_UP = 3     # lookups in a row that must fail before the song is dropped

# --- Whisper as a song-change detector ---
MIN_WORDS = 5            # fewer heard words than this is no evidence either way
EXPECT_WORDS = 8         # lyric words scheduled in the last few seconds = "singing expected"
SILENT_PASSES = 3        # passes with singing expected but none heard -> probably skipped into an intro
SINGING_WORDS = 8        # this many clear words while a song WITHOUT lyrics is up -> maybe a new song
MISMATCH_CHECK = 2       # passes where the words don't fit -> ask Shazam
MISMATCH_DROP = 5        # still not fitting and Shazam couldn't vouch -> hide the lyrics
JUMP_SECONDS = 15.0      # words found this far from where we think we are -> user seeked
ALARM_COOLDOWN = 20.0    # after a false alarm, distrust Whisper on this song for a while

# --- lyric timing calibration ---
CAL_MIN_LINES = 4        # line starts that must agree before shifting the lyrics
CAL_MAX_SPREAD = 0.4     # ...within this (median absolute deviation, seconds)
CAL_MIN_SHIFT = 0.8      # smaller offsets are within normal lyric-file lead time: leave alone
CAL_MAX_SHIFT = 8.0

# --- live band ---
LIVE_REIDENTIFY = 3      # passes without locating the singer before considering other songs
LIVE_GIVE_UP = 15        # passes without locating the singer before dropping the song
GENIUS_COOLDOWN = 15.0
SUSPECT_REASONS = ("gap", "words don't fit", "no singing", "position jump", "recheck")
RECENT = 5.0               # seconds at the end of Whisper's window that must fit the lyrics by themselves
NEW_SONG_GAP_GRACE = 8.0   # a song that was just identified isn't changing again: its quiet intro isn't a gap


class Budget:
    """Token bucket for Shazam lookups."""

    def __init__(self):
        self.tokens, self.stamp = BUCKET_SIZE - 1.0, time.time()

    def available(self) -> float:
        now = time.time()
        self.tokens = min(BUCKET_SIZE, self.tokens + (now - self.stamp) / BUCKET_REFILL)
        self.stamp = now
        return self.tokens

    def take(self):
        self.tokens = max(0.0, self.available() - 1)

    def drain(self):
        self.tokens, self.stamp = 0.0, time.time()


class Engine:
    def __init__(self, source, transcriber, library: Library, *, min_level=0.0005,
                 setlist=None, genius_token=None, use_shazam=True, identify=recognize.identify,
                 translate_to: str | None = None):
        self.source, self.transcriber, self.library = source, transcriber, library
        self.min_level, self.setlist = min_level, setlist
        self.translate_to = translate_to   # show lyrics in this language (None: as they are)
        self.genius_token, self.use_shazam, self.identify = genius_token, use_shazam, identify
        self.clients: set = set()
        self.generation = 0          # bumped whenever the current song changes or stops
        # loudness
        self.noise: float | None = None
        self.smooth: float | None = None
        self.flags: deque = deque(maxlen=PRESENT_OFF)
        self.present = False
        self.present_since = 0.0
        self.music_levels: deque = deque(maxlen=80)
        self.quiet_since: float | None = None
        self.envelope: float | None = None
        self.next_listen = 0.0
        self.suspect: str | None = None   # a sign of a song change seen while lookups were blocked
        # shazam
        self.budget = Budget()
        self.busy = False
        self.last_call = 0.0
        self.blocked_until = 0.0
        self.idle_checks = 0
        self.next_idle_check = 0.0
        self.next_longshot = time.time() + 10  # give the loudness monitor a moment first
        self.longshots = 0
        self.last_gap = 0.0
        self.stopped_at = 0.0
        self.last_genius = 0.0
        self._clear_song()
        self.state: dict = {"status": "listening", "hearing": False}

    # ------------------------------------------------------------------ state
    def _clear_song(self):
        self.generation += 1
        self.mode = None             # "recorded" (Shazam) | "live" (following sung words)
        self.song_key = None
        self.song_text: SongText | None = None
        self.song_lang = None
        self.misses = self.mismatches = self.jumps = self.live_misses = 0
        self.force_check: str | None = None
        self.pending_gap: float | None = None
        self.last_confirmed = time.time()
        self.alarm_cooldown, self.alarm_until = ALARM_COOLDOWN, 0.0
        self.gap_cooldown, self.gap_until = ALARM_COOLDOWN, 0.0
        self.confirms = self.silent_passes = 0
        self.suspect = None
        self.prev_word = None
        self.locked_at = time.time()
        self.cal: dict[int, float] = {}
        self.quiet_since = None
        self.music_levels.clear()

    def _set_lyrics(self, lyr: dict | None):
        self.song_text = SongText.from_lyrics(lyr)
        self.song_lang = matcher.script_language(self.song_text.fulltext) if self.song_text else None

    # ------------------------------------------------------------------ translation (optional)
    def _translated(self, key: str, lyr: dict | None) -> tuple[dict | None, bool]:
        """(lyrics to show, whether a translation still has to be fetched). Uses what the library remembers."""
        if not (self.translate_to and lyr and ("synced" in lyr or "plain" in lyr)):
            return lyr, False
        known = (self.library.songs.get(key) or {}).get("translations", {})
        if self.translate_to not in known:
            return lyr, True
        lines = known[self.translate_to]
        if not lines or len(lines) != len(translate.lines_of(lyr)):
            return lyr, False   # already in that language
        return translate.with_lines(lyr, lines), False

    async def _translate(self, key: str, lyr: dict) -> dict:
        """Translate now (about a second) and remember it. Falls back to the original lyrics."""
        lines = await translate.translate_lines(translate.lines_of(lyr), self.translate_to)
        if lines is None and not translate.last_failed:
            self.library.set_translation(key, self.translate_to, None)
        elif lines:
            self.library.set_translation(key, self.translate_to, lines)
            log.info("lyrics translated to %s", translate.LANGUAGES.get(self.translate_to, self.translate_to))
        return self._translated(key, lyr)[0]

    async def _show_lyrics(self, key: str, lyr: dict | None, gen: int):
        """Put lyrics on screen - translated first if that's switched on."""
        shown, todo = self._translated(key, lyr)
        if todo:
            try:
                shown = await asyncio.wait_for(asyncio.shield(task := asyncio.ensure_future(self._translate(key, lyr))), 5)
            except asyncio.TimeoutError:   # slow: show the original now, swap when it arrives
                task.add_done_callback(lambda t: asyncio.ensure_future(self._swap_in(key, t, gen)))
        if gen != self.generation:
            return
        self.state.update(lyrics=shown, loading=False, translated=self.translate_to if shown is not lyr else None)
        await self.broadcast()

    async def _swap_in(self, key: str, task, gen: int):
        if gen == self.generation and not task.cancelled() and not task.exception():
            orig = (self.library.songs.get(key) or {}).get("lyrics")
            self.state.update(lyrics=task.result(), translated=self.translate_to if task.result() is not orig else None)
            await self.broadcast()

    async def broadcast(self):
        msg = json.dumps(self.state)
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:
                self.clients.discard(ws)

    async def to_listening(self, why: str, music_continues: bool):
        """Clear the screen. `music_continues`: sound is still playing (song ended or was lost),
        so go straight to identifying what's on now; otherwise wait for music to start."""
        log.info("%s -> listening", why)
        now = time.time()
        self._clear_song()
        self.stopped_at = now
        if music_continues:
            self.present, self.present_since = True, now
            self.idle_checks, self.next_idle_check = 0, now + AFTER_GAP
        else:
            self.present = False
            self.flags.clear()
            self.next_longshot = max(self.next_longshot, now + LONGSHOT_EVERY[0])  # we KNOW it just went quiet
        self.state = {"status": "listening", "hearing": self.hearing}
        await self.broadcast()

    @property
    def hearing(self) -> bool:
        """Show "Identifying..." only while we're actively trying; unidentifiable sound
        (TV, talking) falls back to plain "Listening..." after a few attempts."""
        return self.mode is None and self.present and self.idle_checks < 3

    def song_pos(self, wall: float | None = None) -> float | None:
        if self.state.get("pos") is None:
            return None
        return self.state["pos"] + (wall or time.time()) - self.state["wall"]

    # ------------------------------------------------------------------ run
    async def run(self):
        if self.setlist:
            await self.library.load_setlist(self.setlist)
        self.source.start()
        while self.source.available() < 2:
            await asyncio.sleep(0.2)
        self.tasks = [asyncio.create_task(self._guard(self.level_loop, 0.25)),
                      asyncio.create_task(self._guard(self.beat_loop, 1 / 15))]
        try:
            await self._guard(self.tick, STEP)
        finally:   # stopped (settings changed / shutting down)
            for t in self.tasks:
                t.cancel()
            if hasattr(self.source, "stop"):
                self.source.stop()

    async def _guard(self, fn, every: float):
        while True:
            try:
                await fn()
            except Exception:
                log.exception("%s failed", fn.__name__)
            await asyncio.sleep(every)

    # ------------------------------------------------------------------ beat (for the display)
    async def beat_loop(self):
        """Tell the display how hard the music is hitting right now (0..1), so it can pulse along."""
        if (self.mode is None and not self.hearing) or not self.clients:
            return
        x, _ = self.source.latest(0.07)
        fast = audio.rms(x)
        self.envelope = fast if self.envelope is None else 0.93 * self.envelope + 0.07 * fast
        v = min(1.0, max(0.0, fast / (self.envelope + 1e-9) - 1.0))
        if v > 0.1:
            msg = json.dumps({"beat": round(v, 2)})
            for ws in list(self.clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    self.clients.discard(ws)

    # ------------------------------------------------------------------ loudness monitor
    async def level_loop(self):
        try:
            await self.monitor()
        finally:
            self.maybe_shazam()

    async def monitor(self):
        x, _ = self.source.latest(0.5)
        level, now = audio.rms(x), time.time()
        self.smooth = level if self.smooth is None else 0.75 * self.smooth + 0.25 * level
        # Noise floor = the quietest the room has been, creeping up slowly so it can adapt.
        # (Only while nothing plays: during hours of music it would creep up to the music's own quiet
        # passages, and those would then read as "paused".)
        creep = 1.0 if self.mode else 1.0001 if self.present else 1.0005
        self.noise = self.smooth if self.noise is None else min(self.smooth, self.noise * creep)
        self.flags.append(level > max(self.noise * PRESENT_RATIO, self.min_level))

        if self.mode is None:
            before = self.hearing
            recent = list(self.flags)
            if not self.present and len(recent) >= PRESENT_ON and all(recent[-PRESENT_ON:]):
                self.present, self.present_since = True, now - PRESENT_ON * 0.25
                self.idle_checks, self.next_idle_check = 0, self.present_since + AFTER_GAP
                log.info("music started (level %.4f, room %.4f)", level, self.noise)
            elif self.present and len(recent) == PRESENT_OFF and not any(recent):
                self.present = False
            if self.hearing != before:
                self.state = {"status": "listening", "hearing": self.hearing}
                await self.broadcast()
            return

        # Finished: we know how long the track is
        lyr = self.state.get("lyrics") or {}
        pos = self.song_pos()
        if self.mode == "recorded" and pos is not None:
            if lyr.get("track_duration") and pos > lyr["track_duration"] + END_GRACE:
                return await self.to_listening("song finished", music_continues=True)
            if lyr.get("duration") and pos - self.state.get("offset", 0) > lyr["duration"] + 2 * END_GRACE + 1:
                return await self.to_listening("song finished", music_continues=True)

        # Paused / stopped: loudness falls well below the song's usual level
        if len(self.music_levels) < 8:
            self.music_levels.append(level)
            return
        music = float(np.median(self.music_levels))
        if music > 4 * self.noise:
            threshold = PRESENT_RATIO * self.noise   # clearly louder than the room: "paused" = sounds like the empty room
        elif music > 1.5 * self.noise:
            threshold = self.noise + 0.35 * (music - self.noise)  # quiet background music
        else:
            threshold = music * 0.15       # music barely above the room: only a big drop counts
        if level < max(threshold, self.min_level):
            self.quiet_since = self.quiet_since or now
            at_room_level = self.smooth < 1.2 * self.noise
            if now - self.quiet_since >= (STOP_AFTER if at_room_level else STOP_AFTER_SOFT):
                await self.to_listening("music paused/stopped", music_continues=False)
        else:
            self.quiet_since = None
            self.music_levels.append(level)

    # ------------------------------------------------------------------ main loop
    async def tick(self):
        samples, end_wall = self.source.latest(WINDOW)
        start_wall = end_wall - len(samples) / audio.SAMPLE_RATE
        if audio.rms(samples[-int(2 * audio.SAMPLE_RATE):]) < self.min_level:
            return  # dead silence (mic muted / unplugged)

        if self.mode:
            gap = audio.find_gap(samples, self.noise) if time.time() >= self.gap_until else None
            if (gap is not None and start_wall + gap - self.last_gap > 2
                    and start_wall + gap > self.locked_at + 1):  # (the silence before the song isn't a gap in it)
                self.last_gap = self.pending_gap = start_wall + gap
                log.info("gap in the music -> will check for a new song")
                if time.time() < self.blocked_until:
                    await self.suspicion("gap")

        if not self.transcriber:
            return
        if self.mode is None:
            # Only listen to words when there's music Shazam couldn't place (a live band?) and
            # we have lyrics to compare with. Never transcribe a room that's just a room.
            shazam_tried = self.idle_checks >= 1 or time.time() < self.blocked_until or not self.use_shazam
            if not (self.present and shazam_tried and (self.library.texts or self.genius_token)):
                return
        if time.time() < self.next_listen:
            return
        gen, t0 = self.generation, time.time()
        words = await asyncio.to_thread(self.transcriber.words, samples,
                                        self.song_lang if self.mode else None)
        # On a slow machine, listen less often rather than keep the CPU busy: at most a quarter of the time
        took = time.time() - t0
        self.next_listen = time.time() + max(0.0, 4 * took - STEP)
        log.debug("whisper pass took %.2fs", took)
        if gen != self.generation:
            return  # the song changed while we were transcribing
        if words:
            log.debug("heard: …%s", " ".join(w.text for w in words)[-70:])
        if self.mode == "recorded":
            await self.verify(words, start_wall)
            if self.mismatches or self.silent_passes:
                self.next_listen = 0.0   # something's off: listen again right away rather than rest
        else:
            await self.on_words(words, start_wall)

    # ------------------------------------------------------------------ Shazam
    def maybe_shazam(self):
        """Start a lookup if there's a reason to and the budget allows it."""
        now = end_wall = time.time()
        if (not self.use_shazam or self.busy or now < self.blocked_until
                or now - self.last_call < MIN_SPACING):
            return
        reason, urgent, since = None, True, None
        if self.mode is None:
            if self.present:
                if now >= self.next_idle_check and now - self.present_since >= AFTER_GAP:
                    reason, urgent, since = "identify", self.idle_checks < 2, self.present_since
            elif now >= self.next_longshot:
                reason, urgent = "long shot", False
        else:
            if self.pending_gap:
                if end_wall - self.pending_gap < AFTER_GAP:
                    return  # a gap check is coming up: don't waste a lookup on a clip that straddles it
                reason, since = "gap", self.pending_gap
            elif self.force_check:
                reason = self.force_check
            elif self.misses and now - self.last_call >= RETRY_AFTER_MISS:
                reason = "recheck"
            elif now - max(self.last_confirmed, self.last_call) > (
                    UNCONFIRMED_FULL if self.budget.available() >= BUCKET_SIZE - 0.05 else UNCONFIRMED):
                reason, urgent = "unconfirmed", False
        if not reason or self.budget.available() < (1 if urgent else RESERVE + 1):
            return
        # Only audio from after the gap / onset: a clip mixing two songs won't match
        seconds = CLIP if since is None else max(AFTER_GAP, min(CLIP, end_wall - since))
        clip, end_wall = self.source.latest(seconds)
        if len(clip) < AFTER_GAP * audio.SAMPLE_RATE or audio.rms(clip) < self.min_level:
            return  # not enough audio yet / dead silence: don't waste a lookup
        if reason == "gap":
            self.pending_gap = None
        if reason == self.force_check:
            self.force_check = None
        if reason == "long shot":
            self.next_longshot = now + LONGSHOT_EVERY[min(self.longshots, len(LONGSHOT_EVERY) - 1)]
            self.longshots += 1
        self.busy, self.last_call = True, now
        self.budget.take()
        asyncio.create_task(self.shazam(reason, clip, end_wall - len(clip) / audio.SAMPLE_RATE))

    async def shazam(self, reason: str, clip: np.ndarray, clip_start: float):
        t = time.time()
        try:
            match = await asyncio.wait_for(self.identify(audio.to_wav(clip)), TIMEOUT)
        except Exception as e:
            if isinstance(e, (asyncio.TimeoutError, RateLimited)):
                self.budget.drain()
                self.blocked_until = time.time() + BLOCK_PAUSE
                log.warning("shazam (%s): rate limited, pausing lookups %.0fs", reason, BLOCK_PAUSE)
                if self.mode == "recorded" and reason in SUSPECT_REASONS and clip_start >= self.stopped_at:
                    # We had a reason to think the song changed and can't find out for a while. One
                    # alarm is often false, so the lyrics stay - unless a second, different sign
                    # shows up while we wait (see suspicion()).
                    self.suspect = reason
            else:
                log.warning("shazam (%s): error: %s", reason, e)
            return
        finally:
            self.busy = False
        log.info("shazam (%s): %s in %.1fs", reason,
                 f"{match.artist} - {match.title} @ {match.offset:.1f}s" if match else "no match",
                 time.time() - t)
        if clip_start < self.stopped_at:
            return  # that audio is from before the music stopped: stale
        try:
            if match:
                await self.on_match(match, clip_start, reason)
            else:
                await self.on_miss()
        except Exception:
            log.exception("handling shazam result failed")

    async def on_miss(self):
        now = time.time()
        if self.mode is None:
            if self.present:
                self.next_idle_check = now + IDLE_BACKOFF[min(self.idle_checks, len(IDLE_BACKOFF) - 1)]
                was = self.hearing
                self.idle_checks += 1
                if was != self.hearing:
                    self.state = {"status": "listening", "hearing": False}
                    await self.broadcast()
        elif self.mode == "recorded":
            self.misses += 1
            if self.misses >= RECORDED_GIVE_UP:
                await self.to_listening("can't recognize the music any more", music_continues=True)
                self.idle_checks, self.next_idle_check = 1, now + IDLE_BACKOFF[0]

    async def on_match(self, m: recognize.Match, clip_start: float, reason: str):
        now = time.time()
        self.longshots = 0
        key = song_key(m.title, m.artist)
        if key == self.song_key:
            drift = (self.song_pos(clip_start) or m.offset) - m.offset
            if reason in ("words don't fit", "position jump", "no singing") and abs(drift) < 3:
                # Whisper cried wolf: this song is evidently hard for it. Trust it less for a while.
                self.alarm_until = now + self.alarm_cooldown
                self.alarm_cooldown = min(self.alarm_cooldown * 2, 120)
            elif reason == "gap":
                # A quiet moment inside the song, not a track change: this song has those.
                self.gap_until = now + self.gap_cooldown
                self.gap_cooldown = min(self.gap_cooldown * 2, 160)
            self.silent_passes = 0
            self.mode = "recorded"
            self.misses = self.mismatches = self.jumps = 0
            self.last_confirmed = now
            if abs(drift) > 3:
                self.cal.clear()  # the user seeked
            self.state.update(mode="recorded", pos=m.offset, wall=clip_start, line=None)
            return await self.broadcast()

        # A new song: show it at once, lyrics follow as soon as we have them.
        self._clear_song()
        gen = self.generation
        self.mode, self.song_key, self.last_confirmed = "recorded", key, now
        self.locked_at = clip_start
        self.gap_until, self.pending_gap = now + NEW_SONG_GAP_GRACE, None
        entry = self.library.songs.get(key)
        self.state = {"status": "playing", "mode": "recorded", "key": key, "title": m.title,
                      "artist": m.artist, "cover": m.cover, "genre": getattr(m, "genre", None),
                      "pos": m.offset, "wall": clip_start,
                      "lyrics": None, "loading": True,
                      "offset": (entry or {}).get("offset", 0)}
        if entry:
            self._set_lyrics(entry["lyrics"])
            shown, todo = self._translated(key, entry["lyrics"])
            if not todo:
                self.state.update(lyrics=shown, loading=False,
                                  translated=self.translate_to if shown is not entry["lyrics"] else None)
        await self.broadcast()
        if entry:
            if self.state["loading"]:
                await self._show_lyrics(key, entry["lyrics"], gen)
            return
        lyr = await lyrics.fetch(m.title, m.artist, m.adamid)
        if lyr and ("synced" in lyr or "plain" in lyr):
            await self.library.add(m.title, m.artist, lyr)
        if gen != self.generation:
            return  # moved on to another song while fetching
        self._set_lyrics(lyr)
        await self._show_lyrics(key, lyr, gen)

    # ------------------------------------------------------------------ Whisper: recorded song
    async def verify(self, words: list, start_wall: float):
        """Do the words being sung fit the lyrics on screen? Confirms the song without Shazam,
        notices song changes and seeks, and measures whether the lyric timestamps run early/late."""
        heard = [w for w in words if matcher.norm(w.text)]
        st, now = self.song_text, time.time()
        texts = [w.text for w in heard]
        if not (st and st.words):
            # No lyrics on screen (instrumental, or none found). Clear singing may mean the song
            # changed - or just that this song's lyrics are missing; the false-alarm cooldown
            # keeps the latter from costing more than an occasional lookup.
            if len(heard) >= SINGING_WORDS and now >= self.alarm_until:
                self.mismatches += 1
                if self.mismatches == MISMATCH_CHECK:
                    log.info("singing heard during a song without lyrics -> checking")
                    self.force_check = "words don't fit"
            else:
                self.mismatches = 0
            return
        if len(heard) < MIN_WORDS:
            # Little or no singing heard. Fine in an instrumental passage - but if the lyrics say
            # lines are being sung right now, and we could hear this song's singing before, the
            # user has probably skipped to another song's intro.
            if self.singing_expected() and self.confirms >= 2 and now >= self.alarm_until:
                self.silent_passes += 1
                if self.silent_passes >= SILENT_PASSES:
                    log.info("lyrics say singing, but none heard -> checking")
                    self.silent_passes, self.force_check = 0, "no singing"
            return
        self.silent_passes = 0
        expected = None
        pos = self.song_pos()
        if st.times and pos is not None:
            t = pos - self.state.get("offset", 0)
            line = max((i for i, lt in enumerate(st.times) if lt <= t), default=0)
            expected = next((i for i, l in enumerate(st.word_line) if l >= line), None)
        loc = matcher.locate(texts, st, expected)
        # Right after a skip the window holds both songs: its older part still fits the lyrics on
        # screen. So the newest few seconds are judged on their own too.
        recent = [w.text for w in heard if w.start >= WINDOW - RECENT]
        if (loc and MIN_WORDS < len(recent) < len(texts) and not matcher.locate(recent, st, expected)
                and matcher.score_text(" ".join(recent), st) < matcher.WEAK_FIT):
            loc, texts, heard = None, recent, heard[-len(recent):]
        if loc:
            self.mismatches = 0
            self.last_confirmed = now
            self.confirms += 1
            if st.times:
                await self.check_timing(heard, texts, loc[0], start_wall)
            elif self.state.get("line") != st.word_line[loc[0]]:
                # Unsynced lyrics: no timestamps, so show where the singer is instead
                self.state.update(line=st.word_line[loc[0]], line_wall=start_wall + heard[-1].end)
                await self.broadcast()
            return
        self.jumps = 0
        if matcher.score_text(" ".join(texts), st) >= matcher.WEAK_FIT or now < self.alarm_until:
            return
        if not self.confirms:
            return   # Whisper hasn't managed to follow this song at all: its "doesn't fit" means nothing
        self.mismatches += 1
        log.info("words don't fit %s (%d): …%s", self.state.get("title"), self.mismatches,
                 " ".join(texts)[-50:])
        if self.mismatches >= 2 and time.time() < self.blocked_until:
            return await self.suspicion("words don't fit")
        # Normally two passes must disagree before a lookup is spent. With lookups to spare, clear
        # singing that doesn't fit is checked at once: that's the fastest way to catch a skip.
        eager = (self.mismatches == 1 and len(heard) >= SINGING_WORDS and self.confirms >= 3
                 and self.budget.available() >= RESERVE + 1)   # (only when Whisper has followed this song well)
        if self.mismatches == MISMATCH_CHECK or eager:
            self.force_check = "words don't fit"
        elif self.mismatches >= MISMATCH_DROP:
            await self.to_listening("song changed, new one not identified yet", music_continues=True)

    async def suspicion(self, sign: str):
        """A sign of a song change while lookups are blocked. Two different signs: clear the screen -
        an empty screen beats half a minute of the wrong song's lyrics."""
        if time.time() >= self.blocked_until:
            self.suspect = None
        elif self.suspect and self.suspect != sign:
            await self.to_listening("song seems to have changed and it can't be checked yet", music_continues=True)
        else:
            self.suspect = sign

    def singing_expected(self) -> bool:
        st, pos = self.song_text, self.song_pos()
        if not st.times or pos is None:
            return False
        t = pos - self.state.get("offset", 0)
        lines = [i for i, lt in enumerate(st.times) if t - 9 <= lt <= t - 1.5]
        return sum(len(matcher.norm(st.lines[i]).split()) for i in lines) >= EXPECT_WORDS

    async def check_timing(self, heard: list, texts: list[str], end: int, start_wall: float):
        st = self.song_text
        # Seek detection: the words are from a different part of the song than expected
        line, frac = st.word_line[end], st.word_frac[end]
        t1 = st.times[line + 1] if line + 1 < len(st.times) else st.times[line] + 4
        sung_at = st.times[line] + frac * (t1 - st.times[line])
        where = self.song_pos(start_wall + heard[-1].end) - self.state.get("offset", 0)
        if abs(sung_at - where) > JUMP_SECONDS:
            self.jumps += 1
            if self.jumps >= 2:
                self.jumps, self.force_check = 0, "position jump"
            return
        self.jumps = 0
        # Calibration: when a line's first word is heard, compare the clock to its timestamp
        for hi, li in matcher.line_starts(texts, st, end):
            delta = self.song_pos(start_wall + heard[hi].start) - st.times[li]
            if abs(delta) <= CAL_MAX_SHIFT + 2:
                self.cal[li] = delta
        if len(self.cal) < CAL_MIN_LINES:
            return
        vals = np.array(list(self.cal.values()))
        med = float(np.median(vals))
        if float(np.median(np.abs(vals - med))) > CAL_MAX_SPREAD:
            return
        new = 0.0 if abs(med) < CAL_MIN_SHIFT else float(np.clip(med, -CAL_MAX_SHIFT, CAL_MAX_SHIFT))
        if abs(new - self.state.get("offset", 0)) >= 0.3:
            log.info("lyric timing: these lyrics run %.1fs %s for this recording -> shifted",
                     abs(new), "early" if new > 0 else "late")
            self.state["offset"] = round(new, 2)
            self.library.set_offset(self.song_key, round(new, 2))
            await self.broadcast()

    # ------------------------------------------------------------------ Whisper: live band
    async def on_words(self, words: list, start_wall: float):
        heard = [w for w in words if matcher.norm(w.text)]
        if self.mode == "live":
            if heard and self.follow(heard, start_wall):
                self.last_confirmed = time.time()
                return await self.broadcast()
            self.live_misses += 1
            if self.live_misses >= LIVE_GIVE_UP:
                return await self.to_listening("lost the singer", music_continues=True)
            if self.live_misses < LIVE_REIDENTIFY:
                return
        if len(heard) < matcher.MIN_IDENTIFY_WORDS:
            return
        gen = self.generation
        key = await self.identify_text(" ".join(w.text for w in heard))
        if not key or key == self.song_key or gen != self.generation:
            return
        song = self.library.songs[key]
        log.info("[live] recognized from the words: %s - %s", song["artist"], song["title"])
        self._clear_song()
        self.mode, self.song_key = "live", key
        self._set_lyrics(song["lyrics"])
        self.state = {"status": "playing", "mode": "live", "key": key, "title": song["title"],
                      "artist": song["artist"], "cover": None, "lyrics": song["lyrics"],
                      "loading": False, "offset": 0, "line": None, "pos": None, "wall": time.time()}
        shown, todo = self._translated(key, song["lyrics"])
        self.state.update(lyrics=shown, translated=self.translate_to if shown is not song["lyrics"] else None)
        self.follow(heard, start_wall)
        await self.broadcast()
        if todo:   # (the band plays on: translate in the background and swap the lines in)
            task = asyncio.ensure_future(self._translate(key, song["lyrics"]))
            task.add_done_callback(lambda t, g=self.generation: asyncio.ensure_future(self._swap_in(key, t, g)))

    def follow(self, heard: list, start_wall: float) -> bool:
        """Locate the singer in the current song's lyrics and re-anchor the display."""
        st = self.song_text
        loc = matcher.locate([w.text for w in heard], st, self.prev_word)
        if not loc:
            return False
        wi, score = loc
        line, frac = st.word_line[wi], st.word_frac[wi]
        self.prev_word, self.live_misses = wi, 0
        update = {"line": line, "wall": start_wall + heard[-1].end, "pos": None}
        if st.times:  # estimate song time of the last heard word, for smooth progress
            t0 = st.times[line]
            t1 = st.times[line + 1] if line + 1 < len(st.times) else t0 + 4
            update["pos"] = t0 + frac * (t1 - t0)
        self.state.update(update)
        log.info("[live] line %d (score %.0f): %s", line, score, st.lines[line])
        return True

    async def identify_text(self, text: str) -> str | None:
        found = matcher.identify(text, self.library.texts)
        if found:
            return found[0]
        if self.genius_token and time.time() - self.last_genius >= GENIUS_COOLDOWN:
            self.last_genius = time.time()
            try:
                hits = await genius.search(" ".join(text.split()[-12:]), self.genius_token)
            except Exception as e:
                log.warning("genius error: %s", e)
                return None
            for title, artist in hits:
                key = await self.library.add(title, artist)
                if key and self.library.texts[key].words and \
                        matcher.score_text(text, self.library.texts[key]) >= matcher.IDENTIFY_THRESHOLD - 5:
                    return key
        return None
