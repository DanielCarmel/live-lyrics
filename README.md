# Live Lyrics

An ambient lyrics display for a bar or a living room. It listens to whatever music is playing in the
room, works out which song it is, and shows the lyrics full-screen, in time with the music, with animated
text that changes style with the genre. It is something to look at, not a karaoke machine.

- **Hands-off.** Start it, play music. Songs appear within about 5 seconds, switch when the music switches,
  and clear when it pauses or ends.
- **Recorded music and live bands.** Recordings are recognised by sound; a band is recognised by the words
  it sings.
- **Any language** that has lyrics available, including right-to-left scripts such as Hebrew.
- **Optional translation** of every song into English, Hebrew or Spanish.
- **Free.** No accounts or API keys (one optional free key, for Genius).
- **Runs on modest hardware.** It uses a graphics card when there is one and scales itself down when
  there isn't.

## Contents
- [Quick start](#quick-start)
- [What you see](#what-you-see)
- [Docker](#docker)
- [Settings](#settings)
- [How it works](#how-it-works)
- [Command-line options](#command-line-options)
- [Privacy and outside services](#privacy-and-outside-services)
- [Known limits](#known-limits)
- [Project layout](#project-layout)

## Quick start

Needs Python 3.12+, [uv](https://docs.astral.sh/uv/), a microphone, and Chrome/Chromium (or Firefox) for
the full-screen window. On Linux the microphone library also needs PortAudio (`sudo apt install libportaudio2`).

```bash
uv sync                           # once
uv run live-lyrics --port 8000    # opens the start page full-screen
```

Pick a translation language (or leave it off), press **Start**, and play music.

> The default port is 80, which ordinary users can't open on Linux outside Docker - hence `--port 8000`
> above. With Docker the default works as is.

The first start downloads a Whisper speech model (150-500 MB) into `./models`.

## What you see

| Screen | When |
|---|---|
| **Start page** | On launch. Choose settings, press Start. With saved settings it starts by itself after 15 s |
| **Listening** | No music recognised. A ring of bars around an orb of light; it stirs when it hears music |
| **Now playing** | A song was just recognised: cover art, title, artist |
| **Lyrics** | One line at a time fills the screen and pops in with its own move, then floats gently |
| **Break** | Intros and instrumental passages show a picture that moves with the beat |

**Looks.** There are 26 text styles, each with its own typeface (and a matching Hebrew one), entrance
moves and break picture: spinning record with the cover art, level meter, heartbeat trace, neon rings,
bouncing dots, flowing waves, music notes, or an equaliser. The genre reported for the song picks a family
of four to six styles (pop, rock, hip-hop, dance, soul, indie, latin, jazz/classical) and the song itself
picks one of them, so the same song always looks the same and two songs in a row usually don't. Colours
come from the cover art.

**Keys on the lyrics screen**

| Key | |
|---|---|
| `S` or `Esc` | Stop and go back to the start page |
| `↑` / `↓` | Shift lyric timing by 0.25 s (remembered) |
| `0` | Reset the timing shift |
| `F` | Toggle full-screen |

## Docker

The container does the listening and serves the pages; any browser is the screen.

```bash
docker compose up -d --build
```

Then open **http://localhost** (port 80) in a browser, or full-screen:

```bash
google-chrome --kiosk http://localhost
```

| Task | Command |
|---|---|
| Watch what it hears and recognises | `docker compose logs -f` |
| Stop | `docker compose down` |
| Start again | `docker compose up -d` |
| Rebuild after changing the code | `docker compose up -d --build` |

### What the compose file sets up

- **Microphone.** It hands the container this machine's PulseAudio/PipeWire socket, so the container hears
  the desktop's default microphone. The container runs as your user (`UID`/`GID`, default 1000) because
  the sound socket only admits its owner.
- **No sound server** (a headless box): remove the `PULSE_SERVER` line and the pulse volume, and pass the
  sound card instead:
  ```yaml
      devices: ["/dev/snd"]
      group_add: ["audio"]
  ```
- **Kept between runs.** `./data` (settings, the song library) and `./models` (Whisper models) are mounted
  from the project folder.
- **Restarts.** `restart: unless-stopped` brings the container back after a reboot. An open lyrics screen
  notices, returns to the start page and starts again with the saved settings.
- **Unattended start.** Add `command: ["--start"]` to begin at once with the settings used last time,
  without anyone opening the start page. Flags can follow, e.g. `["--start", "--translate", "he"]`.

### Network access

The compose file publishes port 80 on all network interfaces, **with no password**. Anyone who can reach
the page can see what is playing, change the settings and stop the app. They cannot hear the microphone:
audio never leaves the machine.

- **This machine only:** change the port line to `"127.0.0.1:80:80"`.
- **Require a password:** add `LIVE_LYRICS_PASSWORD: "something-long"` under `environment:`. Every browser
  then signs in once and stays signed in for a year.
- **From the internet:** forward a port on the router to this machine's port 80. The connection is plain
  HTTP, so a password would travel unencrypted; for anything beyond a quick experiment, put it behind HTTPS
  (a reverse proxy, or a tunnel such as Cloudflare Tunnel or Tailscale).

### Image variants

| Image | Build | Size | Notes |
|---|---|---|---|
| CPU (default) | `docker build -t live-lyrics .` | about 0.9 GB | Works everywhere. Uses the lighter Whisper models |
| NVIDIA graphics card | `docker build -t live-lyrics:gpu --build-arg GPU=1 .` | about 3.2 GB | Run with `--gpus all`. Builds, but has not been tested running |

Inside a container the app sees no graphics card (unless the GPU image is used), so it picks the lighter
Whisper model. The lyrics screen is drawn by whichever browser opens it, with that machine's graphics.

## Settings

**On the start page**

| Setting | Choices |
|---|---|
| Translate the lyrics | Off, English, Hebrew, Spanish |
| Microphone *(More options)* | System default or any input device found |
| Recognise recordings *(More options)* | Shazam on or off |
| Live band *(More options)* | A setlist file, and a Genius token |

Choices are saved in `data/settings.json` and offered again next time.

**Decided automatically from the machine**

| Machine | Look | Following the singing |
|---|---|---|
| Graphics card (NVIDIA/CUDA) | Full effects | Whisper `small` |
| No graphics card, 8+ CPU cores | Light | Whisper `base` |
| No graphics card, 4-7 cores | Light | Whisper `tiny` |
| No graphics card, under 4 cores | Light | Off |

The light look keeps the styles and animations but drops the blurred cover-art background, floating lights
and slow zoom. Any lyrics screen also switches to it by itself if it can't hold its frame rate. On a slow
CPU, Whisper listens less often rather than keep a core busy.

**Translation.** When on, each song is translated once as it is first recognised (about a second), line for
line so the timing still fits, and the result is remembered. Songs already in the chosen language are shown
as they are. Translations are literal: meaning, not poetry.

## How it works

```
mic ──► loudness monitor (4×/s) ── is there music? did it pause? did the track end?
   ├──► Shazam (free, rate-limited) ── which recording, and where in it
   └──► Whisper (local, unlimited) ── do the sung words still fit? → song changes, timing, live bands
                    │
      lyrics (+ optional translation) + track length ──► WebSocket ──► browser display
```

- **Loudness monitor.** Tracks the room's noise floor. Nothing is sent to Shazam and nothing is transcribed
  unless sound stands clearly above it. It also detects a pause (the level falls back to room level) and
  the end of a track (the position passes the track's known length).
- **Shazam** (via [`shazamio`](https://github.com/shazamio/ShazamIO)) identifies recordings and returns the
  exact position in the song. The free service blocks after a few lookups in quick succession, so lookups
  are budgeted and made only for a reason: music started, a blip of silence (a skip), the words stopped
  fitting, the lyrics say singing but none is heard, or nothing has confirmed the song for a while.
- **Whisper** ([faster-whisper](https://github.com/SYSTRAN/faster-whisper), running locally) transcribes the
  singing every couple of seconds while a song is up. That is how the app, without asking Shazam:
  - confirms the song is still the same (the words fit the lyrics on screen);
  - notices a song change or a seek (the words don't fit, or fit another part);
  - corrects lyric timing: if a lyric file consistently runs early or late against what is heard, the
    display is shifted and the correction remembered;
  - follows live bands, which Shazam can't match: the song is recognised from its words (a setlist, every
    song seen before, optionally Genius) and the display follows the singer line by line.
- **Lyrics**, best source first: [LRCLIB](https://lrclib.net) synced → NetEase Cloud Music synced → LRCLIB
  plain → lyrics.ovh plain. A candidate must match on title plus artist or exact track length (from
  Apple's free lookup), which avoids most wrong-version matches. Lyrics without timestamps still move:
  they follow the singer via Whisper, with an estimate in between.
- **Display.** A web page fed over a WebSocket. Text styles, entrances and break pictures are plain CSS
  animations; the beat value from the microphone drives the break pictures and the background glow.

## Command-line options

Run with no options to use the start page. Giving any setting on the command line skips the start page.

| Flag | |
|---|---|
| `--start` | Skip the start page and begin with the settings used last time (plus any flags given) |
| `--translate LANG` | Show lyrics translated into `en`, `he` or `es` |
| `--no-gpu` | Use no graphics card: Whisper on the CPU with a smaller model, light look |
| `--host` / `--port` | Address and port to serve on (default `0.0.0.0`, `80`) |
| `--no-display` | Don't open a browser window; open the address in any browser yourself |
| `--device N` | Microphone, by index or name (`--list-devices` shows them) |
| `--setlist FILE` | `Artist - Title` per line. For live bands: lyrics are fetched ahead and matched by the words |
| `--genius-token T` / env `GENIUS_TOKEN` | Free token from https://genius.com/api-clients, to recognise live songs outside the setlist (untested) |
| `--whisper-model` | `tiny` / `base` / `small` / `medium` / `large-v3` (default: chosen from the machine) |
| `--whisper-threads N` | CPU cores Whisper may use when on the CPU (default 2) |
| `--language he` | Force the sung language (default: from the lyrics, else detected) |
| `--no-live` | No Whisper: recordings only, slower song-change detection |
| `--no-shazam` | No Shazam: live mode only |
| `--verbose` | Also log what Whisper hears (off by default: it may include room conversation) |
| `--identify FILE` | Identify an audio file and exit |
| `--simulate FILE` | Feed an audio file in real time instead of the microphone, for testing (needs ffmpeg) |

Environment: `LIVE_LYRICS_PASSWORD` turns on the sign-in page.

## Privacy and outside services

- **Audio stays on the machine**, with one exception: short clips (about 6 seconds) are sent to Shazam to
  identify the song, and only while music is detected.
- **Whisper runs locally.** What it hears is not stored, and not logged unless `--verbose` is given.
- **Sent out as text:** song title and artist (to the lyric sources and Apple's lookup), and the lyrics
  themselves when translation is on (to Google Translate).
- **The display loads fonts from Google Fonts** and cover art from Apple's image servers.
- **Shazam and Google Translate are used through unofficial, keyless access.** They can change or be
  limited without notice; this is a personal project, not something to build a product on as is.
- **Stored locally** in `data/`: settings (including a Genius token, in plain text), the song library with
  lyrics, timing corrections and translations, and the browser profile of the full-screen window.

## Known limits

- **Shazam's free limit is unpredictable.** When it blocks, lookups pause for 30 seconds; songs already
  known are still recognised from their words meanwhile, new ones wait.
- **A quiet microphone signal hurts everything.** If the music is barely above the room's noise, quiet
  passages look like pauses and sung words are hard to make out. Put the microphone near a speaker, or
  feed it from the sound system's line-out.
- **Skipping straight into an instrumental stretch**, with no blip of silence, has no local tell-tale; it is
  caught by the periodic double-check, typically within 15-30 seconds.
- **The lighter Whisper models** (used without a graphics card) are weaker on sung words, especially in
  Hebrew, so song changes without a pause are caught more slowly there.
- **Songs whose title is listed differently** by Shazam and the lyric sources (for example only a
  transliterated title) may show the title card without lyrics.
- **Live mode needs vocals** and a microphone that hears them clearly.
- **Crowd noise** over the music (a busy bar) has not been tested.

## Project layout

```
live_lyrics/
  app.py          web server, start page API, command line
  engine.py       the brain: when to look up, song changes, pauses, timing, live bands
  audio.py        microphone ring buffer, loudness, gap detection
  recognize.py    Shazam
  transcribe.py   Whisper
  lyrics.py       lyric sources and matching
  translate.py    optional translation
  matcher.py      fitting heard words to lyrics
  library.py      remembered songs (data/library-v2.json)
  genius.py       optional lyric search for live songs
  static/
    setup.html    start page
    index.html    listening screen and lyrics display
Dockerfile, docker-compose.yml, docker/entrypoint.sh
```
