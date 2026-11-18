"""Local speech-to-text of sung vocals with faster-whisper (free, offline)."""
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("live-lyrics")

# Whisper's classic hallucinations on music / noise / silence
_JUNK_ANYWHERE = ("thank you for watching", "thanks for watching", "subtitles by", "amara.org",
                  "please subscribe", "transcribed by", "end of the video", "see you next time",
                  "see you in the next")
_JUNK_WHOLE = {"thank you", "thanks", "you", "bye", "okay", "ok", "yeah", "oh", "so", "the end"}


@dataclass
class Word:
    text: str
    start: float  # seconds relative to the start of the audio passed in
    end: float


class Transcriber:
    def __init__(self, model: str = "auto", device: str = "auto",
                 language: str | None = None, models_dir: str = "models", cpu_threads: int = 2):
        """model "auto": `small` on a graphics card, `base` on the CPU (about a third of the work).
        On the CPU, Whisper is kept to `cpu_threads` cores so the display stays smooth."""
        _ensure_writable_hf_home(models_dir)
        from faster_whisper import WhisperModel
        import ctranslate2

        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        auto, cpu = model == "auto", dict(device="cpu", compute_type="int8", cpu_threads=cpu_threads,
                                          download_root=models_dir)
        if auto:
            model = "small" if device == "cuda" else "base"
        try:
            self.model = WhisperModel(model, **cpu) if device == "cpu" else WhisperModel(model, device=device,
                                      compute_type="float16" if device == "cuda" else "int8",
                                      download_root=models_dir)
        except Exception as e:
            if device == "cpu":
                raise
            log.warning("Whisper on %s failed (%s), using CPU", device, e)
            device, model = "cpu", "base" if auto else model
            self.model = WhisperModel(model, **cpu)
        self.language = language
        self.device = device
        log.info("Whisper '%s' loaded on %s", model, device)

    def words(self, samples: np.ndarray, language: str | None = None) -> list[Word]:
        """Words sung/spoken in `samples`. Returns [] rather than guesses when the
        transcript looks unreliable (hallucinations on instrumental music or noise)."""
        language = language or self.language
        segments, info = self.model.transcribe(
            samples, language=language, beam_size=1, word_timestamps=True,
            condition_on_previous_text=False, vad_filter=False)
        if language is None and info.language_probability < 0.5:
            return []  # can't even tell the language: not clear singing
        out = []
        for s in segments:
            text = s.text.strip().lower()
            bare = text.strip(" .!?,…")
            if (s.avg_logprob < -1.0 or s.compression_ratio > 2.4 or s.no_speech_prob > 0.85
                    or bare in _JUNK_WHOLE or any(j in text for j in _JUNK_ANYWHERE)):
                continue
            out += [Word(w.word.strip(), w.start, w.end) for w in (s.words or []) if w.word.strip()]
        return out


def _ensure_writable_hf_home(models_dir: str):
    """Hugging Face writes caches and logs under $HF_HOME. If that isn't writable, use ours."""
    home = os.environ.get("HF_HOME")
    if home and not os.access(home, os.W_OK):
        os.environ["HF_HOME"] = str(Path(models_dir, "hf").resolve())
