"""
Transcriber: merges faster-whisper segments with pyannote diarization.
Optimized for low VRAM environments via sequential model loading and offloading.
"""

import logging
import math
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Optional, Generator

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch

from config import Settings

logger = logging.getLogger(__name__)

_PYANNOTE_CHUNK_SAMPLES = 160_000


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


class Transcriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Initializing asynchronous-ready transcriber architecture...")

    def _flush_gpu(self):
        """Forces Python to clean up variables and release unused VRAM cache back to the OS."""
        import gc
        gc.collect()
        if self.settings.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ──────────────────────────────────────────
    def run(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None,
    ) -> TranscriptionResult:

        # --- PHASE 1: WHISPER TRANSCRIPTION ---
        logger.info("Loading Whisper model '%s' dynamically to GPU…", self.settings.whisper_model)
        whisper_model = WhisperModel(
            self.settings.whisper_model,
            device=self.settings.device,
            device_index=self.settings.device_index,
            compute_type=self.settings.compute_type,
            num_workers=self.settings.whisper_num_workers,
            cpu_threads=self.settings.whisper_cpu_threads,
        )

        try:
            logger.info("Transcribing '%s'…", audio_path)
            whisper_segments, info = whisper_model.transcribe(
                audio_path,
                language=language,
                beam_size=self.settings.beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                word_timestamps=True,
            )

            # Materialize generator while Whisper is fully active on GPU
            raw_segments = list(whisper_segments)
            
        finally:
            # Explicitly delete the whisper reference and sweep VRAM
            logger.info("Offloading Whisper model from GPU memory...")
            del whisper_model
            self._flush_gpu()

        if not raw_segments:
            return TranscriptionResult(
                duration=info.duration,
                language=info.language,
                language_probability=info.language_probability,
            )

        # --- PHASE 2: PYANNOTE DIARIZATION ---
        logger.info("Loading pyannote diarization pipeline dynamically to GPU…")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=self.settings.hf_token,
        )

        if self.settings.device == "cuda":
            diarization_pipeline = diarization_pipeline.to(
                torch.device(f"cuda:{self.settings.device_index}")
            )

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

            logger.info("Running speaker diarization…")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*resulted in \d+ samples instead of.*",
                    category=UserWarning,
                )
                diarization = diarization_pipeline(padded_path, **diarize_kwargs)

            turns = [
                (round(turn.start, 3), round(turn.end, 3), speaker)
                for turn, speaker in diarization.speaker_diarization
            ]

        finally:
            # Explicitly delete the diarizer reference and sweep VRAM
            logger.info("Offloading Pyannote pipeline from GPU memory...")
            del diarization_pipeline
            self._flush_gpu()
            
            if is_temp:
                os.unlink(padded_path)

        # --- PHASE 3: MERGING RESULTS ---
        merged = self._merge(raw_segments, turns)

        return TranscriptionResult(
            duration=info.duration,
            language=info.language,
            language_probability=info.language_probability,
            segments=merged,
        )

    # ──────────────────────────────────────────
    @staticmethod
    def _pad_audio(audio_path: str) -> tuple[str, bool]:
        data, sr = sf.read(audio_path, always_2d=False, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)

        if sr != 16_000:
            try:
                import resampy
                data = resampy.resample(data, sr, 16_000)
            except ImportError:
                from scipy.signal import resample_poly
                g = math.gcd(sr, 16_000)
                data = resample_poly(data, 16_000 // g, sr // g).astype("float32")
            sr = 16_000

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
        result: list[Segment] = []
        for ws in whisper_segments:
            seg_start, seg_end = ws.start, ws.end
            best_speaker = "UNKNOWN"
            best_overlap = 0.0

            for t_start, t_end, speaker in turns:
                overlap = max(0.0, min(seg_end, t_end) - max(seg_start, t_start))
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