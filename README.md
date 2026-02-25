# 🎙️ Audio Transcription API

FastAPI service that transcribes audio **and** identifies individual speakers.

```
POST /transcribe  →  {full_text, segments: [{speaker, start, end, text}], …}
```

---

## Stack & why these choices

| Component | Choice | Reason |
|---|---|---|
| Transcription | `faster-whisper medium` (int8) | ~4× faster than openai-whisper, half the RAM, near-identical accuracy |
| Diarization | `pyannote/speaker-diarization-3.1` | State-of-the-art, free on HuggingFace |
| Framework | FastAPI | Async, auto-docs, typed |
| Device | auto (CUDA → CPU) | Runs anywhere, GPU gives ~8× speedup |

**Sweet spot:** `medium` model at `int8` gives you ~95% of `large-v2` accuracy at 3× the speed and half the VRAM.

---

## Setup

### 1. HuggingFace token (required for diarization)

1. Create a free account at https://huggingface.co
2. Accept the license for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept the license for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Generate a token at https://huggingface.co/settings/tokens

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **GPU users:** swap the `torch` line in `requirements.txt` for the CUDA build.

### 3. Configure

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=hf_xxxxxxxxxx
```

Or just export it:
```bash
export HF_TOKEN=hf_xxxxxxxxxx
```

### 4. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

First request will download models (~1.5 GB). Pre-warm with:
```bash
curl -X POST http://localhost:8000/warm-up
```

---

## API

Interactive docs: **http://localhost:8000/docs**

### `POST /transcribe`

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file (mp3/wav/m4a/ogg/flac/webm) |
| `language` | string | auto | ISO 639-1 code, e.g. `en` |
| `num_speakers` | int | auto | Exact speaker count |
| `min_speakers` | int | auto | Min speakers hint |
| `max_speakers` | int | auto | Max speakers hint |

**Example:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@interview.mp3" \
  -F "max_speakers=3"
```

**Response:**
```json
{
  "duration_seconds": 124.5,
  "language": "en",
  "language_probability": 0.9991,
  "num_speakers": 2,
  "full_text": "[SPEAKER_00] Hello, welcome to the show.\n[SPEAKER_01] Thanks for having me.",
  "segments": [
    { "speaker": "SPEAKER_00", "start": 0.0,  "end": 2.1,  "text": "Hello, welcome to the show." },
    { "speaker": "SPEAKER_01", "start": 2.8,  "end": 5.4,  "text": "Thanks for having me." }
  ],
  "processing_time_seconds": 18.3
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "device": "cpu", "compute_type": "int8" }
```

---

## Tuning

| Goal | Change |
|---|---|
| **Faster, less accurate** | `WHISPER_MODEL=small`, `beam_size=1` |
| **Slower, more accurate** | `WHISPER_MODEL=large-v3`, `beam_size=5` |
| **GPU acceleration** | Install CUDA torch build; `DEVICE=cuda` auto-detected |
| **Known speaker count** | Pass `num_speakers=N` in the request |

---

## Project structure

```
audio_transcriber/
├── main.py           # FastAPI routes
├── transcriber.py    # Whisper + pyannote merge logic
├── config.py         # Settings (env-based)
├── requirements.txt
└── README.md
```
