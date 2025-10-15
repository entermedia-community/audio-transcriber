from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile, os
from faster_whisper import WhisperModel

os.environ["OMP_NUM_THREADS"] = "4"

app = FastAPI()
model = WhisperModel("base.en", device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
  try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
      tmp.write(await file.read())
      tmp_path = tmp.name

    segments, _ = model.transcribe(audio=tmp_path, vad_filter=True, language="en", log_progress=True)
    os.remove(tmp_path)
    
    json = []
    for _, seg in enumerate(segments, start=1):
      start = seg.start
      end = seg.end
      text = seg.text.strip()
      json.append({
        "start": start,
        "end": end,
        "text": text
      })
    return JSONResponse(content=json)
  except Exception as e:
    return JSONResponse(content={"error": str(e)}, status_code=500)

