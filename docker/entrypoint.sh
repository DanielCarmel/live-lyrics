#!/bin/sh
# Sound input: with a PulseAudio/PipeWire socket handed in (PULSE_SERVER), make it ALSA's default
# device; otherwise ALSA talks to the sound card passed with --device /dev/snd.
set -e
mkdir -p "$HOME"
if [ -n "$PULSE_SERVER" ]; then
  printf 'pcm.!default pulse\nctl.!default pulse\n' > "$HOME/.asoundrc"
fi
exec live-lyrics --host 0.0.0.0 --no-display "$@"
