# Live Lyrics

An ambient lyrics display for a bar or a living room: it listens to whatever music is playing, identifies
the song, and shows its lyrics full-screen with animations, in time with the music. It is something to
look at, not a karaoke machine. Works for **recorded music** (speakers, radio, DJ) and **live bands**.
No interaction needed.

**Cost: $0.** No accounts or API keys (one optional free key, for Genius).

```bash
uv sync                 # once
uv run live-lyrics      # opens the full-screen display and starts listening
```

Play music: within ~5 s the song appears and the lyrics follow it. Pause, and they fade out. Skip to another
song, and they switch. Ctrl+C stops it and closes the window.

## How it works

```
mic ──► loudness monitor (4×/s) ── is there music? did it pause? did the track end?
   ├──► Shazam (free, rate-limited) ── which recording, and where in it
   └──► Whisper (local, unlimited) ── do the sung words still fit? → song changes, timing, live bands
                    │
          LRCLIB lyrics + Apple track length ──► WebSocket ──► browser display
```

- **Loudness monitor** (`engine.py: monitor`). Tracks the room's noise floor. Nothing is sent to Shazam and
  nothing is transcribed unless sound stands clearly above it. Also detects pause/stop (level falls back to
  room level for 2 s) and song end (position passes the track's known length).
- **Shazam** via [`shazamio`](https://github.com/shazamio/ShazamIO) identifies recordings and returns the
  exact position. The free endpoint blocks after a few lookups in quick succession, so lookups are budgeted
  (token bucket) and made only for a reason: music started, a blip of silence (a skip), the words stopped
  fitting, the lyrics say singing but none is heard, or nothing has confirmed the song for a while.
- **Whisper** ([faster-whisper](https://github.com/SYSTRAN/faster-whisper), local) transcribes the singing
  every ~2 s while a song is up. It is used to:
  - confirm the song without asking Shazam (words fit the lyrics on screen);
  - detect a song change or seek (words don't fit, or fit a different part);
  - **calibrate lyric timing**: when the first word of a line is heard, its timestamp is compared with the
    clock. If the lyric file consistently runs early/late (a different version of the track), the display is
    shifted, and the correction is remembered for next time;
  - follow **live bands**, which Shazam can't match: the song is recognized from its words (setlist + every
    song seen before, optionally Genius) and the display follows the singer line by line.
- **Lyrics**, best source first: [LRCLIB](https://lrclib.net) synced → NetEase Cloud Music synced →
  LRCLIB plain → lyrics.ovh plain. Candidates are gathered for several title variants (including
  `Transliterated (מקור)` forms) and must match on title plus artist or exact track length (from Apple's
  free lookup API), which avoids most wrong-version and wrong-song matches. Unsynced lyrics still move:
  they follow the singer via Whisper, with a gentle auto-scroll estimate in between.
- **Display** (`static/index.html`): Listening → Identifying… → "now playing" card → lyrics. One line at a
  time fills the screen and pops in with its own move (slam, bounce, glitch, typewriter, wave...). The
  look - typeface, moves, idle motion - is picked from the song's genre (Shazam's), and varied per song,
  so rock, hip-hop, dance, soul, indie, latin, jazz and pop each feel different. There are 26 looks in all:
  each genre has four to six, and the song decides which one it gets. Intros and instrumental breaks show
  the look's own picture (equaliser, level meter, spinning record with the cover art, neon rings, bouncing
  dots, flowing waves, music notes or a heartbeat trace), moving with the beat. The glow and background
  pulse with the music and take their colour from the cover art; intros and breaks show an equaliser.
  The listening screen is a slowly moving orb of light that stirs when it hears music. Fonts come from
  Google Fonts (system fonts are used offline). If the machine can't hold the frame rate (no graphics
  card), the page switches itself to a lighter look: no blurred cover, floating lights or slow zoom. Right-to-left scripts supported; keeps the screen awake.

**Start page.** `uv run live-lyrics` opens a start page full-screen where every setting is a button:
translation, and under "More options" the microphone, Shazam, a setlist file and a Genius token. The look
(full effects / light) and how closely the singing is followed are set automatically from the machine: a
graphics card gets everything; without one, the light look and a Whisper model sized to the CPU. Press **Start**; on the lyrics screen
press **S** (or Esc) to come back and change something. The choices are remembered in `data/settings.json`;
next time the page starts with them by itself after 15 seconds unless someone touches it. The flags below
still work and skip the page, as does `--start` (begin at once with last time's settings).

## Docker
The container does the listening and serves the pages; any browser is the screen.

```bash
docker compose up -d --build      # then open http://localhost:80
```

- **Sound**: `docker-compose.yml` hands the container this machine's PulseAudio/PipeWire socket, so it hears
  the default microphone. On a box without a sound server, pass the sound card instead (see the comments in
  the file).
- **Kept between runs**: `./data` (settings, song library) and `./models` (Whisper, downloaded on first start).
- **Unattended**: add `command: ["--start"]` to begin at once with the settings used last time.
- **Network / internet access**: the compose file publishes port 80 to the network with no password, so
  anyone who can reach the page can also change the settings and stop it. To require a password, add
  `LIVE_LYRICS_PASSWORD: "..."` under `environment:`; every browser then signs in once. The connection is
  plain HTTP either way.
- **Graphics card** (optional, NVIDIA, untested): `docker build -t live-lyrics:gpu --build-arg GPU=1 .` and
  run with `--gpus all`. Adds about 2.3 GB. The default image is CPU-only (about 0.9 GB).
- For a full-screen window, open the page in a kiosk browser, e.g. `google-chrome --kiosk http://localhost:80`.

## Options
| Flag | |
|---|---|
| `--translate LANG` | Show the lyrics translated into `en`, `he` or `es` (off unless given). Each song is translated once with Google's free web translation and remembered; songs already in that language are shown as they are |
| `--no-gpu` | Use no graphics card: Whisper on the CPU with the smaller `base` model (2 cores, at most about a third of the time), display drawn in software |
| `--setlist FILE` | `Artist - Title` per line. For live bands: lyrics are prefetched and matched offline |
| `--genius-token T` / env `GENIUS_TOKEN` | Free token from https://genius.com/api-clients, to identify live songs not in the setlist (untested) |
| `--language he` | Force the sung language (default: from the lyrics' script, else auto-detect) |
| `--whisper-model` | `tiny` / `base` / `small` (default) / `medium` / `large-v3` |
| `--no-live` | No Whisper: recorded music only, slower song-change detection |
| `--no-display` | Don't open the window (view from another device with `--host 0.0.0.0`) |
| `--device N` | Mic (`--list-devices`) |
| `--verbose` | Also log what Whisper hears (off by default: it may include room conversation) |
| `--simulate FILE` | Feed an audio file in real time instead of the mic (testing) |

Display keys: `↑`/`↓` nudge timing ±0.25 s (remembered), `0` reset, `f` fullscreen.

## Known limits
- **Shazam's free limit is unpredictable.** If it blocks us, lookups pause 30 s; songs already known are still
  recognized from their words meanwhile, new ones wait.
- **Songs with no usable title in LRCLIB** (e.g. Shazam reports only a transliterated title while LRCLIB has
  the native one) show a title card without lyrics.
- **Very quiet music** (less than ~1.5× the room's noise) isn't detected as "music started"; a rare background
  lookup still finds it, within a minute or so.
- **Skipping into a wordless stretch with no blip of silence** (instrumental → another song's instrumental
  intro) has no local tell-tale; it's caught by the periodic double-check, typically 15-30 s. Sound-based
  change detection (timbre, key) was tried and was too unreliable to be worth its false alarms.
- **Live mode needs vocals** and a mic that hears them clearly.
- Whisper `small` needs a GPU for comfort (~0.3 s per pass); on CPU it works but lags (~3 s per pass).
