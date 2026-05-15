import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.graph import run_research

PAPERS_DIR  = Path(__file__).parent.parent / "papers"
STATIC_DIR  = Path(__file__).parent / "static"

app = FastAPI(title="ResearchCrew")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config():
    def short(model_id: str) -> str:
        return model_id.split("/")[-1] if model_id else "—"
    return {
        "hype_model":       short(os.getenv("HYPE_MODEL", "")),
        "contrarian_model": short(os.getenv("CONTRARIAN_MODEL", "")),
        "arbitro_model":    short(os.getenv("ARBITRO_MODEL", "")),
        "writer_model":     short(os.getenv("WRITER_MODEL", "")),
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/research")
async def research(
    question: str = Form(...),
    pdfs: List[UploadFile] = File(...),
):
    # Validar y guardar PDFs en papers/
    pdf_paths = []
    for pdf in pdfs:
        if not pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Solo se aceptan archivos PDF. Archivo rechazado: {pdf.filename}",
            )
        content = await pdf.read()
        path = PAPERS_DIR / pdf.filename
        path.write_bytes(content)
        pdf_paths.append(str(path))

    # run_research es síncrono (bloquea) — se ejecuta en threadpool
    result = await run_in_threadpool(run_research, question, pdf_paths)

    return JSONResponse(content=dict(result))
