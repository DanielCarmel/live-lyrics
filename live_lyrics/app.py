"""Web server + command line. The logic lives in engine.py."""
import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import numpy as np
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from . import audio, lyrics, recognize
from .engine import Engine
from .library import Library

log = logging.getLogger("live-lyrics")
STATIC = Path(__file__).parent / "static"


BROWSER_FLAGS: list[str] = []   # extra Chrome flags (--no-gpu adds --disable-gpu)


def launch_display(url: str):
    """Open the lyrics page full-screen (kiosk) with no clicks needed."""
    profile = Path("data/browser-profile").resolve()  # own profile: kiosk works even if Chrome is already open
    for exe in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        if path := shutil.which(exe):
            cmd = [path, f"--app={url}", "--kiosk", "--start-fullscreen", f"--user-data-dir={profile}",
                   "--no-first-run", "--noerrdialogs", "--disable-infobars",
                   "--disable-session-crashed-bubble", "--password-store=basic", *BROWSER_FLAGS]
            break
    else:
        if not (path := shutil.which("firefox")):
            log.warning("no browser found; open %s manually", url)
            return None
        cmd = [path, "--kiosk", "--new-instance", "--profile", str(profile), url]
    profile.mkdir(parents=True, exist_ok=True)
    log.info("opening display: %s", url)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Lyrics</title><style>
html{background:#06060a}body{min-height:100vh;margin:0;display:flex;align-items:center;justify-content:center;padding:16px;color:#f5f5f7;
font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:radial-gradient(60% 60% at 30% 30%,rgba(124,156,255,.22),transparent 70%)}
form{width:min(380px,100%);display:flex;flex-direction:column;gap:14px}h1{font-size:.95rem;letter-spacing:.42em;font-weight:800;color:rgba(255,255,255,.6);margin:0}
h1 em{font-style:normal;color:#7c9cff}label{font-size:1.5rem;font-weight:800}p{margin:0;color:#ff8f8f}
input,button{font:inherit;padding:.8em 1em;border-radius:12px;border:1px solid rgba(255,255,255,.15)}input{background:rgba(0,0,0,.35);color:#fff}
button{border:0;font-weight:800;color:#0a0a12;background:linear-gradient(100deg,#7c9cff,#c77dff 55%,#ff7ab8);cursor:pointer}
input:focus-visible,button:focus-visible{outline:2px solid #fff;outline-offset:2px}</style></head><body>
<form method="post" action="/login"><h1>LIVE <em>LYRICS</em></h1><label for="p">Password</label><!--msg-->
<input id="p" name="password" type="password" autofocus autocomplete="current-password" required><button>Open</button></form></body></html>"""

DEFAULTS = {"translate": None, "light": None, "whisper": "auto", "language": None, "device": None,
            "shazam": True, "setlist": "", "genius_token": ""}
SETTINGS_FILE = Path("data/settings.json")


def load_settings() -> dict | None:
    """What was chosen on the start page last time (None: never)."""
    try:
        # (look, Whisper model and song language are never remembered: the machine / the song decide)
        return {**DEFAULTS, **json.loads(SETTINGS_FILE.read_text()), "light": None, "whisper": "auto", "language": None}
    except Exception:
        return None


def has_gpu() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def auto_mode() -> dict:
    """What this machine can run comfortably: the look, and how closely to follow the singing.
    A graphics card: everything. Without one: the light look, and a Whisper model sized to the CPU."""
    if has_gpu():
        return {"light": False, "whisper": "small"}
    cores = os.cpu_count() or 1
    return {"light": True, "whisper": "base" if cores >= 8 else "tiny" if cores >= 4 else "off"}


def microphones() -> list[str]:
    try:
        import sounddevice as sd
        return list(dict.fromkeys(d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0))
    except Exception:
        return []


class Session:
    """The running lyrics engine, started (and restarted) with the settings picked on the start page."""

    def __init__(self, *, whisper_threads: int = 2, whisper_device: str = "auto", simulate=None, port: int = 80):
        self.engine: Engine | None = None
        self.task: asyncio.Task | None = None
        self.settings: dict | None = None
        self.clients: set = set()
        self.mic_client: WebSocket | None = None   # the browser that is the microphone, if any
        self.transcribers: dict = {}
        self.browser = None
        self.kiosk = False   # we open the full-screen window ourselves
        self.whisper_threads, self.whisper_device, self.simulate, self.port = whisper_threads, whisper_device, simulate, port

    def display_path(self) -> str:
        # The light look is forced only on a window we opened ourselves without the graphics card.
        # Any other browser (another machine, a container's host) finds out by itself if it needs it.
        return "/display?lite=1" if (self.settings or {}).get("light") and (self.browser or self.kiosk) else "/display"

    async def start(self, settings: dict):
        settings = {**DEFAULTS, **settings}
        auto = auto_mode()   # (unless forced from the command line, the machine decides)
        if settings["light"] is None:
            settings["light"] = auto["light"]
        if settings["whisper"] == "auto":
            settings["whisper"] = "base" if settings["light"] and auto["whisper"] == "small" else auto["whisper"]
        if settings["setlist"] and not Path(settings["setlist"]).expanduser().is_file():
            raise ValueError(f"setlist file not found: {settings['setlist']}")
        await self.stop()
        if self.simulate:
            source = audio.FileSource(*self.simulate)
        elif settings["device"] == "browser" or (not settings["device"] and not microphones()):
            source = audio.BrowserMic()   # no sound card in reach (e.g. Docker on a Mac): the browser listens for us
        else:
            dev = settings["device"]
            source = audio.Mic(int(dev) if isinstance(dev, str) and dev.isdigit() else dev or None)
        transcriber = None
        if settings["whisper"] != "off":
            from .transcribe import Transcriber
            device = "cpu" if settings["light"] else self.whisper_device
            key = (settings["whisper"], device)
            if key not in self.transcribers:   # loading takes a few seconds: keep models that were used before
                self.transcribers[key] = await asyncio.to_thread(
                    Transcriber, settings["whisper"], device, cpu_threads=self.whisper_threads)
            transcriber = self.transcribers[key]
            transcriber.language = settings["language"]
        self.engine = Engine(source, transcriber, Library(), setlist=settings["setlist"] or None,
                             genius_token=settings["genius_token"] or None, use_shazam=settings["shazam"],
                             translate_to=settings["translate"])
        self.engine.clients = self.clients
        self.settings = settings
        self.task = asyncio.create_task(self.engine.run())
        log.info("started: %s", ", ".join(f"{k}={v}" for k, v in settings.items() if k != "genius_token" and v not in (None, "")))

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except BaseException:
                pass
        self.engine = self.task = None

    def mic(self) -> str:
        return "browser" if self.engine and isinstance(self.engine.source, audio.BrowserMic) else "server"

    async def show(self, path: str, light: bool):
        """Open (or re-open) the full-screen window. Drawing without the graphics card is a browser
        start-up choice, so switching it means a fresh window."""
        if self.browser:
            self.browser.terminate()
            await asyncio.to_thread(self.browser.wait)
        BROWSER_FLAGS[:] = ["--disable-gpu"] if light else []
        self.browser = launch_display(f"http://0.0.0.0:{self.port}{path}")
        self.browser_light = light


def build_app(session, open_path: str | None = None, start_with: dict | None = None) -> FastAPI:
    """`open_path`: page to open full-screen once the server is up. `start_with`: settings to start
    with straight away (otherwise the start page asks)."""
    if isinstance(session, Engine):   # (tests hand over a ready-made engine)
        engine, session = session, Session()
        session.engine, session.clients, session.settings = engine, engine.clients, dict(DEFAULTS)
        session.task = "external"

    @asynccontextmanager
    async def lifespan(_):
        if session.task == "external":
            session.task = asyncio.create_task(session.engine.run())
        elif start_with is not None:
            await session.start(start_with)

        async def open_when_ready():
            while True:  # wait until the server accepts connections
                try:
                    _, w = await asyncio.open_connection("0.0.0.0", session.port)
                    w.close()
                    break
                except OSError:
                    await asyncio.sleep(0.2)
            await session.show(open_path, bool((session.settings or load_settings() or {}).get("light")))

        opener = asyncio.create_task(open_when_ready()) if open_path else None
        yield
        await session.stop()
        if opener:
            opener.cancel()
        if session.browser:
            session.browser.terminate()

    app = FastAPI(lifespan=lifespan)
    page = lambda name: FileResponse(STATIC / name, headers={"Cache-Control": "no-store"})

    # Password (LIVE_LYRICS_PASSWORD): needed as soon as the page is reachable by anyone else, because
    # whoever can open it can also change the settings and stop it. One sign-in per browser.
    password = os.environ.get("LIVE_LYRICS_PASSWORD") or ""
    ticket = hmac.new(os.urandom(32) if not password else hashlib.sha256(("live-lyrics|" + password).encode()).digest(),
                      b"ticket", "sha256").hexdigest()
    allowed = lambda cookies: not password or hmac.compare_digest(cookies.get("ll", ""), ticket)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        if allowed(request.cookies) or request.url.path == "/login":
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "sign in first"}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    @app.get("/login")
    async def login_page():
        return HTMLResponse(LOGIN_PAGE)

    @app.post("/login")
    async def login(request: Request):
        given = parse_qs((await request.body()).decode(errors="replace")).get("password", [""])[0]
        if not password or not hmac.compare_digest(given.encode(), password.encode()):
            await asyncio.sleep(1.5)   # slows down guessing
            return HTMLResponse(LOGIN_PAGE.replace("<!--msg-->", "<p>That password isn't right.</p>"), status_code=401)
        out = RedirectResponse("/", status_code=303)
        out.set_cookie("ll", ticket, max_age=365 * 86400, httponly=True, samesite="lax")
        return out

    @app.get("/")
    async def index():
        return page("index.html" if session.engine else "setup.html")

    @app.get("/display")
    async def display():
        return page("index.html")

    @app.get("/setup")
    async def setup():
        return page("setup.html")

    @app.get("/api/options")
    async def options():
        saved = load_settings()
        token_saved = bool((saved or {}).get("genius_token"))
        return {"settings": {**DEFAULTS, **(saved or {}), "genius_token": ""}, "saved": saved is not None,
                "token_saved": token_saved,   # (the token itself never leaves the server)
                "running": session.engine is not None, "auto": auto_mode(), "microphones": microphones()}

    @app.post("/api/start")
    async def start(settings: dict):
        settings = {k: settings.get(k, v) for k, v in DEFAULTS.items()}
        settings.update(light=None, whisper="auto", language=None)   # not choices: decided per machine / per song
        if not settings["genius_token"]:   # left empty on the page: keep the one already saved
            settings["genius_token"] = (load_settings() or {}).get("genius_token", "")
        try:
            await session.start(settings)
        except Exception as e:
            log.exception("could not start")
            return JSONResponse({"error": str(e)}, status_code=400)
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=1))
            tmp.replace(SETTINGS_FILE)
        except OSError as e:
            log.warning("could not save the settings: %s", e)
        light = bool(session.settings["light"])
        if session.browser and getattr(session, "browser_light", light) != light:
            asyncio.get_running_loop().call_later(0.3, lambda: asyncio.ensure_future(session.show(session.display_path(), light)))
            return {"reopening": True, "mic": session.mic()}
        return {"url": session.display_path(), "mic": session.mic()}

    @app.post("/api/stop")
    async def stop():
        await session.stop()
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        if not allowed(sock.cookies):
            return await sock.close(code=4401)
        await sock.accept()
        session.clients.add(sock)
        await sock.send_text(json.dumps({**session.engine.state, "mic": session.mic()} if session.engine else {"status": "setup"}))
        try:
            while True:
                await sock.receive_text()
        except WebSocketDisconnect:
            session.clients.discard(sock)

    @app.websocket("/ws/audio")
    async def ws_audio(sock: WebSocket):
        """A browser's microphone: a text frame {"rate": Hz} first, then binary float32 PCM frames."""
        if not allowed(sock.cookies):
            return await sock.close(code=4401)
        if session.mic() != "browser":
            return await sock.close(code=4404)
        await sock.accept()
        if session.mic_client:   # one browser listens at a time: the newest
            try:
                await session.mic_client.close(code=4409)
            except Exception:
                pass
        session.mic_client = sock
        rate = audio.SAMPLE_RATE
        log.info("microphone: browser connected (%s)", sock.client.host if sock.client else "?")
        frames, loud = 0, 0.0
        try:
            while True:
                msg = await sock.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text"):
                    rate = int(json.loads(msg["text"]).get("rate", audio.SAMPLE_RATE))
                    log.info("microphone: browser sends %d Hz", rate)
                elif msg.get("bytes") and session.mic() == "browser":   # (survives a restart with new settings)
                    session.engine.source.rate = rate
                    session.engine.source.feed(msg["bytes"])
                    frames += 1
                    loud = max(loud, audio.rms(np.frombuffer(msg["bytes"], dtype=np.float32)))
                    if frames % 100 == 0 and frames <= 300:   # the first half minute: is anything coming in?
                        log.info("microphone: %d s of sound from the browser, peak level %.4f%s", frames // 10, loud,
                                 "" if loud > 0.001 else " (silence: is the microphone allowed and unmuted?)")
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            log.info("microphone: browser disconnected")
            if session.mic_client is sock:
                session.mic_client = None

    return app


async def identify_file(path: str):
    m = await recognize.identify(path)
    if not m:
        print("No match.")
        return
    print(f"{m.artist} - {m.title}  (clip starts at {m.offset:.1f}s)")
    lyr = await lyrics.fetch(m.title, m.artist, m.adamid)
    if not lyr or not ("synced" in lyr or "plain" in lyr):
        print("No lyrics found on LRCLIB.")
    elif "synced" in lyr:
        print(f"{len(lyr['synced'])} synced lines, e.g.:")
        for l in lyr["synced"][:5]:
            print(f"  [{l['t']:6.2f}] {l['text']}")
    else:
        print(f"Plain (unsynced) lyrics, {len(lyr['plain'])} lines")


def main():
    p = argparse.ArgumentParser(description="Live music -> live lyrics display. Run without options to choose the settings on screen.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=80)
    p.add_argument("--device", default=None, help="input device index or name (see --list-devices)")
    p.add_argument("--no-display", action="store_true", help="don't open a browser window; open the address in any browser yourself (servers, Docker)")
    p.add_argument("--verbose", action="store_true", help="also log what Whisper hears")
    p.add_argument("--start", action="store_true",
                   help="skip the start page and begin at once with the settings used last time (plus any flags given)")
    p.add_argument("--translate", metavar="LANG", choices=["en", "he", "es"],
                   help="show the lyrics translated into this language: en (English), he (Hebrew) or es (Spanish). "
                        "Off unless given. Uses Google's free web translation")
    p.add_argument("--no-gpu", action="store_true",
                   help="use no graphics card at all: Whisper on the CPU (smaller model), display drawn in software")
    g = p.add_argument_group("live band mode")
    g.add_argument("--setlist", help="text file, one `Artist - Title` per line, to recognize by lyrics")
    g.add_argument("--genius-token", default=os.environ.get("GENIUS_TOKEN"),
                   help="free Genius API token for identifying songs not in the setlist (env GENIUS_TOKEN)")
    g.add_argument("--whisper-model", default="auto",
                   help="tiny/base/small/medium/large-v3 (default: small on a graphics card, base on the CPU)")
    g.add_argument("--whisper-threads", type=int, default=2, help="CPU cores Whisper may use when on the CPU")
    g.add_argument("--whisper-device", default="auto", help="auto/cuda/cpu")
    g.add_argument("--language", default=None, help="force sung language, e.g. en, he (default: auto)")
    g.add_argument("--no-live", action="store_true", help="disable Whisper (no live bands, slower song-change detection)")
    g.add_argument("--no-shazam", action="store_true", help="disable Shazam; live mode only")
    t = p.add_argument_group("testing")
    t.add_argument("--list-devices", action="store_true")
    t.add_argument("--identify", metavar="FILE", help="identify an audio file with Shazam and exit")
    t.add_argument("--simulate", metavar="FILE", help="use an audio file played in real time instead of the mic")
    t.add_argument("--simulate-filter", metavar="AF",
                   help='ffmpeg filter to disguise the file as a live take, e.g. "atempo=1.06,asetrate=44100*1.03,aresample=44100"')
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "aiohttp", "shazamio", "shazamio_core", "asyncio",
                  "faster_whisper", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if a.identify:
        asyncio.run(identify_file(a.identify))
        return

    with socket.socket() as probe:  # fail fast and clearly instead of after loading the model
        if probe.connect_ex(("0.0.0.0", a.port)) == 0:
            sys.exit(f"Live Lyrics is already running (port {a.port} is in use). "
                     f"Its display is at http://0.0.0.0:{a.port}")

    # Flags given on the command line (or no screen to click on): start right away, as before.
    # Otherwise the start page asks - and offers last time's choices.
    flags = {"translate": a.translate, "light": True if a.no_gpu else None,
             "whisper": "off" if a.no_live else a.whisper_model, "language": a.language, "device": a.device,
             "shazam": not a.no_shazam, "setlist": a.setlist or "", "genius_token": a.genius_token or ""}
    has_screen = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and not a.no_display
    direct = (a.start or a.simulate
              or any(flags[k] != DEFAULTS[k] for k in flags if k != "genius_token"))
    session = Session(whisper_threads=a.whisper_threads, whisper_device=a.whisper_device, port=a.port,
                      simulate=(a.simulate, a.simulate_filter) if a.simulate else None)
    start_with = ({**(load_settings() or DEFAULTS), **{k: v for k, v in flags.items() if v != DEFAULTS[k]}}
                  if direct else None)
    if start_with and start_with["light"] is None:
        start_with["light"] = auto_mode()["light"]
    session.kiosk = has_screen
    print(f"Live Lyrics: open http://0.0.0.0:{a.port} in a browser"
          + ("" if direct else " to choose the settings and start") + "  (Ctrl+C to stop)", flush=True)
    open_path = None if not has_screen else ("/display?lite=1" if start_with["light"] else "/display") if direct else "/setup"
    app = build_app(session, open_path, start_with)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
