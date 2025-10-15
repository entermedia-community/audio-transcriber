from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile, os
from faster_whisper import WhisperModel, BatchedInferencePipeline

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

app = FastAPI()
model = WhisperModel("base.en", device="cpu", compute_type="int8")
batched_model = BatchedInferencePipeline(model=model)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
  try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
      tmp.write(await file.read())
      tmp_path = tmp.name

    segments, _ = batched_model.transcribe(audio=tmp_path, vad_filter=True, language="en", batch_size=16, chunk_length=5)
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

