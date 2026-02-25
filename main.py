"""
Audio Transcription API with Speaker Diarization
FastAPI + faster-whisper + pyannote.audio
Sweet spot: int8 quantized Whisper + pyannote/speaker-diarization-3.1
"""

import os
import time
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from transcriber import Transcriber, TranscriptionResult
from config import Settings

import librosa
import numpy as np
import io
# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

settings = Settings()

app = FastAPI(
    title="🎙️ Audio Transcription API",
    description=(
        "Transcribe audio with per-speaker labeling.\n\n"
        "**Stack:** faster-whisper (int8 · medium) + pyannote speaker diarization\n\n"
        "Supported formats: mp3, wav, m4a, ogg, flac, webm"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared transcriber instance (lazy-loaded on first request)
_transcriber: Optional[Transcriber] = None

TARGET_LENGTH = 480000
TARGET_SR = 16000  # change if needed

def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        logger.info("Initializing transcriber (first request — this may take a moment)…")
        _transcriber = Transcriber(settings)
    return _transcriber


# ─────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────
class SpeakerSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    duration_seconds: float
    language: str
    language_probability: float
    num_speakers: int
    full_text: str
    segments: list[SpeakerSegment]
    processing_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    compute_type: str


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Check API and model status."""
    loaded = _transcriber is not None
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        device=settings.device,
        compute_type=settings.compute_type,
    )


@app.post("/transcribe", response_model=TranscriptionResponse, tags=["Transcription"])
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(
        None,
        description="ISO 639-1 language code (e.g. 'en'). Auto-detected when omitted.",
    ),
    min_speakers: Optional[int] = Form(None, description="Minimum expected speakers"),
    max_speakers: Optional[int] = Form(None, description="Maximum expected speakers"),
    num_speakers: Optional[int] = Form(
        None, description="Exact number of speakers (overrides min/max)"
    ),
):
    """
    Upload an audio file and receive a full transcript with per-speaker labels.

    - Speaker labels are `SPEAKER_00`, `SPEAKER_01`, etc.
    - Segments are ordered chronologically.
    - `full_text` concatenates all segments with speaker prefixes.
    """
    ALLOWED = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED)}",
        )

    t0 = time.perf_counter()

    # Write upload to a temp file so pyannote / ffmpeg can read it
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcriber = get_transcriber()
        result: TranscriptionResult = transcriber.run(
            audio_path=tmp_path,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers,
        )
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    processing_time = round(time.perf_counter() - t0, 2)

    full_text = "\n".join(
        f"[{seg.speaker}] {seg.text}" for seg in result.segments
    )

    return TranscriptionResponse(
        duration_seconds=round(result.duration, 2),
        language=result.language,
        language_probability=round(result.language_probability, 4),
        num_speakers=len({s.speaker for s in result.segments}),
        full_text=full_text,
        segments=[
            SpeakerSegment(
                speaker=seg.speaker,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text.strip(),
            )
            for seg in result.segments
        ],
        processing_time_seconds=processing_time,
    )


@app.post("/warm-up", tags=["Meta"])
def warm_up():
    """Pre-load the model into memory so the first transcription is fast."""
    get_transcriber()
    return {"status": "models loaded", "device": settings.device}
