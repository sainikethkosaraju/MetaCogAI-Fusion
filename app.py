"""MetaCogAI web app.

    uvicorn app:app --reload        # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from metacogai.system import MetaCogAI

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="MetaCogAI", description="A multimodal classifier that knows when it doesn't know.")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
system = MetaCogAI()
MAX_BYTES = 10 * 1024 * 1024


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(image: UploadFile | None = File(None), text: str = Form("")):
    pil = None
    if image is not None and image.filename:
        data = await image.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image larger than 10 MB.")
        try:
            pil = Image.open(io.BytesIO(data))
            pil.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "That file is not an image I can read.")
    try:
        return system.analyze(pil, text).to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e))
