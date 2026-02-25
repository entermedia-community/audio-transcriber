"""
Transcriber: merges faster-whisper segments with pyannote diarization.

Pipeline:
1. faster-whisper  → timestamped word/segment transcription
2. pyannote        → speaker turn annotations  (SPEAKER_00, SPEAKER_01 …)
3. merge           → assign each whisper segment the dominant speaker

Fix: audio is zero-padded to a multiple of pyannote's 10-second chunk
(160 000 samples @ 16 kHz) before diarization.  This prevents the noisy
"requested N samples … resulted in M samples" UserWarning that fires when
the last window is shorter than expected.
"""

import logging
import math
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch

from config import Settings

logger = logging.getLogger(__name__)

# pyannote chunks audio into 10-second windows at 16 kHz  →  160 000 samples
_PYANNOTE_CHUNK_SAMPLES = 160_000


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────
@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    duration: float
    language: str
    language_probability: float
    segments: list[Segment] = field(default_factory=list)


# ─────────────────────────────────────────────
# Transcriber
# ─────────────────────────────────────────────
class Transcriber:
    def __init__(self, settings: Settings):
        self.settings = settings

        logger.info(
            "Loading Whisper model '%s' on %s (%s)…",
            settings.whisper_model,
            settings.device,
            settings.compute_type,
        )
        self.whisper = WhisperModel(
            settings.whisper_model,
            device=settings.device,
            compute_type=settings.compute_type,
            num_workers=settings.whisper_num_workers,
            cpu_threads=settings.whisper_cpu_threads,
        )

        logger.info("Loading pyannote diarization pipeline…")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=settings.hf_token,
        )

        if settings.device == "cuda":
            self.diarization_pipeline = self.diarization_pipeline.to(
                torch.device("cuda")
            )

        logger.info("Models ready.")

    # ──────────────────────────────────────────
    def run(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None,
    ) -> TranscriptionResult:

        # 1. Transcribe ──────────────────────────
        logger.info("Transcribing '%s'…", audio_path)
        whisper_segments, info = self.whisper.transcribe(
            audio_path,
            language=language,
            beam_size=self.settings.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True,
        )

        # Materialise the lazy generator before diarization starts
        raw_segments = list(whisper_segments)

        if not raw_segments:
            return TranscriptionResult(
                duration=info.duration,
                language=info.language,
                language_probability=info.language_probability,
            )

        # 2. Diarise ─────────────────────────────
        logger.info("Running speaker diarization…")

        # Pad audio to a chunk boundary so pyannote never receives a truncated
        # last window — fixes "requested N samples, got M samples" warning.
        padded_path, is_temp = self._pad_audio(audio_path)
        try:
            diarize_kwargs: dict = {}
            if num_speakers:
                diarize_kwargs["num_speakers"] = num_speakers
            else:
                if min_speakers:
                    diarize_kwargs["min_speakers"] = min_speakers
                if max_speakers:
                    diarize_kwargs["max_speakers"] = max_speakers

            # Belt-and-suspenders: also suppress any residual warning from
            # pyannote sub-models that do their own internal chunking.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*resulted in \d+ samples instead of.*",
                    category=UserWarning,
                )
                diarization = self.diarization_pipeline(padded_path, **diarize_kwargs)
        

            # Build a sorted list of (start, end, speaker) turns
            turns = [
                (round(turn.start, 3), round(turn.end, 3), speaker)
                for turn, speaker in diarization.speaker_diarization
            ]

            # 3. Merge ───────────────────────────────
            merged = self._merge(raw_segments, turns)

            return TranscriptionResult(
                duration=info.duration,
                language=info.language,
                language_probability=info.language_probability,
                segments=merged,
            )
        finally:
            if is_temp:
                os.unlink(padded_path)

    # ──────────────────────────────────────────
    @staticmethod
    def _pad_audio(audio_path: str) -> tuple[str, bool]:
        """
        Read audio, downmix to mono 16 kHz, then zero-pad to the nearest
        multiple of _PYANNOTE_CHUNK_SAMPLES.

        Returns (path, is_temp).  Caller must os.unlink(path) when is_temp.
        """
        data, sr = sf.read(audio_path, always_2d=False, dtype="float32")

        # Downmix to mono
        if data.ndim == 2:
            data = data.mean(axis=1)

        # Resample to 16 kHz if needed
        if sr != 16_000:
            try:
                import resampy
                data = resampy.resample(data, sr, 16_000)
            except ImportError:
                from scipy.signal import resample_poly
                g = math.gcd(sr, 16_000)
                data = resample_poly(data, 16_000 // g, sr // g).astype("float32")
            sr = 16_000

        # Zero-pad to the next chunk boundary
        remainder = len(data) % _PYANNOTE_CHUNK_SAMPLES
        if remainder != 0:
            data = np.concatenate(
                [data, np.zeros(_PYANNOTE_CHUNK_SAMPLES - remainder, dtype="float32")]
            )

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, data, sr)
        tmp.close()
        return tmp.name, True

    # ──────────────────────────────────────────
    @staticmethod
    def _merge(whisper_segments, turns: list[tuple]) -> list[Segment]:
        """
        Assign each Whisper segment the speaker with the maximum time overlap.
        Falls back to 'UNKNOWN' if no diarization turn overlaps at all.
        """
        result: list[Segment] = []

        for ws in whisper_segments:
            seg_start, seg_end = ws.start, ws.end
            best_speaker = "UNKNOWN"
            best_overlap = 0.0

            for t_start, t_end, speaker in turns:
                overlap = max(
                    0.0, min(seg_end, t_end) - max(seg_start, t_start)
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker

            result.append(
                Segment(
                    speaker=best_speaker,
                    start=seg_start,
                    end=seg_end,
                    text=ws.text,
                )
            )

        return result