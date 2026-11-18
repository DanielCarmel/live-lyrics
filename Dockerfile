# Live Lyrics: listens to the room, shows the lyrics. The container runs the listening and the web
# page; any browser (on this machine or the network) is the screen.
#
#   docker build -t live-lyrics .                      CPU image (works everywhere)
#   docker build -t live-lyrics:gpu --build-arg GPU=1 .  adds the NVIDIA libraries Whisper needs (run with --gpus all)
FROM python:3.12-slim-bookworm

# libportaudio2: microphone access. libasound2-plugins: lets ALSA use a PulseAudio/PipeWire socket.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libportaudio2 libasound2-plugins \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1

# dependencies first, so editing the code doesn't reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY live_lyrics ./live_lyrics
RUN uv sync --frozen --no-dev

ARG GPU=0
RUN if [ "$GPU" = "1" ]; then uv pip install --python .venv/bin/python "nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9,<10"; fi

COPY docker/entrypoint.sh /entrypoint.sh
# data/ (settings, song library) and models/ (Whisper, downloaded on first start) are worth keeping: mount them.
RUN mkdir -p /app/data /app/models /tmp/home && chmod 777 /app/data /app/models /tmp/home
ENV PATH=/app/.venv/bin:$PATH HOME=/tmp/home PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
VOLUME ["/app/data", "/app/models"]
EXPOSE 80
ENTRYPOINT ["/entrypoint.sh"]
