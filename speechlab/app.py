import asyncio
import json
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from demos import acoustics, articulation, calibration, directions, embeddings, fixtures, providers

from . import audio, guided, jobs, live, models, store
from .catalog import DEMOS
from .config import MAX_BYTES, ROOT

app = FastAPI(title="Speech Lab", version="0.1.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if (
        origin
        and origin != f"{request.url.scheme}://{request.headers.get('host')}"
        and request.method != "GET"
    ):
        return JSONResponse({"detail": "Cross-origin writes are disabled"}, status_code=403)
    return await call_next(request)


@app.exception_handler(ValueError)
async def value_error(_request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(FileNotFoundError)
async def missing(_request, _exc):
    return JSONResponse({"detail": "Record not found"}, status_code=404)


class ClipMeta(BaseModel):
    label: str = Field(default="Untitled recording", max_length=150)
    target: str = Field(default="", max_length=300)
    language: str = Field(default="zh-CN", max_length=30)
    speaker: str = Field(default="me", max_length=80)
    session: str = Field(default="session-1", max_length=80)
    role: Literal["attempt", "reference", "anchor", "fixture"] = "attempt"
    notes: str = Field(default="", max_length=2000)
    region_start: float = Field(default=0, ge=0)
    region_end: float | None = Field(default=None, gt=0)


class Analysis(BaseModel):
    demo: Literal[
        "acoustics",
        "embeddings",
        "omni",
        "directions",
        "articulation",
        "calibration",
        "providers",
        "feedback",
    ]
    clip_id: str | None = None
    clip_ids: list[str] = Field(default_factory=list, max_length=40)
    model: str = "wavlm-base"
    layer: int | None = None
    ceiling: int = Field(default=5500, ge=3500, le=8000)
    name: str = Field(default="My contrast", max_length=150)
    negative_label: str = Field(default="Less", max_length=100)
    positive_label: str = Field(default="More", max_length=100)
    negative_ids: list[str] = Field(default_factory=list, max_length=40)
    positive_ids: list[str] = Field(default_factory=list, max_length=40)
    reference_ids: list[str] = Field(default_factory=list, max_length=40)
    anchor_ids: list[str] = Field(default_factory=list, max_length=40)
    axis_id: str | None = None
    group_by: Literal["session", "speaker"] = "session"
    provider: Literal["azure", "speechsuper", "qwen"] = "azure"
    reference: str = Field(default="", max_length=1000)
    language: str = Field(default="zh-CN", max_length=30)
    cloud_consent: bool = False
    condition: Literal["visual", "hidden"] = "visual"


@app.get("/api/status")
def status():
    return {
        "demos": DEMOS,
        "models": models.MODELS,
        "providers": providers.status(),
        "device": models.device(),
        "clip_count": len(store.records("clips")),
    }


class GuidedAnalysis(BaseModel):
    contrast: str
    clip_id: str | None = None
    example_side: int | None = Field(default=None, ge=0, le=1)
    example_index: int = Field(default=0, ge=0, le=2)


@app.get("/api/guided")
def guided_catalog():
    return guided.catalog()


@app.get("/api/live")
def live_catalog():
    return live.catalog()


@app.get("/api/live/{ident}/example/{phone}/{index}")
def live_example(ident: str, phone: str, index: int):
    return FileResponse(live.reference(ident, phone, index), media_type="audio/flac")


@app.post("/api/live/{ident}/frame")
async def live_frame(ident: str, request: Request):
    import numpy as np

    live.get_map(ident)
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > live.WINDOW_SAMPLES * 4:
            raise HTTPException(413, "A live frame must be exactly 160 ms")
    if len(payload) != live.WINDOW_SAMPLES * 4:
        raise ValueError("A live frame must be exactly 160 ms of float32 PCM")
    wave = np.frombuffer(payload, dtype="<f4").copy()
    # Share the single model worker with recorded analyses; model swaps cannot race.
    return await asyncio.get_running_loop().run_in_executor(jobs.POOL, live.frame, ident, wave)


@app.get("/api/guided/{ident}/example/{side}/{index}/{kind}")
def guided_example(ident: str, side: int, index: int, kind: str):
    return FileResponse(guided.reference(ident, side, index, kind), media_type="audio/flac")


@app.post("/api/guided/analyze")
def guided_analyze(body: GuidedAnalysis):
    guided.contrast(body.contrast)
    example = None
    if body.example_side is not None:
        if body.clip_id:
            raise ValueError("Choose an example or a recording, not both.")
        example = (body.example_side, body.example_index)
        guided.reference(body.contrast, *example)
    elif body.clip_id:
        store.read("clips", body.clip_id)
    else:
        raise ValueError("Record a sound first.")
    return jobs.submit(lambda progress: guided.analyze(body.contrast, body.clip_id, example, progress))


@app.get("/api/clips")
def clips():
    return store.records("clips")


@app.post("/api/clips")
async def upload(file: UploadFile = File(...), metadata: str = Form("{}")):
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "Audio file exceeds 25 MB")
    try:
        meta = ClipMeta.model_validate_json(metadata)
    except Exception:
        raise HTTPException(422, "Invalid recording metadata")
    return audio.add(content, meta.model_dump())


@app.patch("/api/clips/{ident}")
def edit_clip(ident: str, meta: ClipMeta):
    clip = store.read("clips", ident)
    # Validate crop before saving it, without resampling or altering original audio.
    audio.load(ident, meta.region_start, meta.region_end)
    clip.update(meta.model_dump())
    store.write_json(store.path_for("clips", ident), clip)
    return clip


@app.get("/api/clips/{ident}/audio")
def get_audio(ident: str):
    return FileResponse(store.path_for("clips", ident, "wav"), media_type="audio/wav")


@app.post("/api/fixtures")
def seed_fixtures():
    return fixtures.seed()


@app.get("/api/axes")
def axes():
    return [{k: v for k, v in a.items() if k not in ("mean", "scale", "coef")} for a in store.records("axes")]


@app.get("/api/results")
def results():
    return [
        {
            k: r[k]
            for k in ("id", "created", "kind", "model", "provider", "clip_id", "name", "condition")
            if k in r
        }
        for r in store.records("results")
    ]


@app.get("/api/results/{ident}")
def result(ident: str):
    return store.read("results", ident)


@app.get("/api/export")
def export():
    payload = {
        "clips": store.records("clips"),
        "axes": store.records("axes"),
        "results": store.records("results"),
    }
    return Response(
        json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="speech-lab-session.json"'},
    )


@app.post("/api/analyze")
def analyze(body: Analysis):
    if body.model not in models.MODELS:
        raise ValueError("Unknown model")
    if body.demo == "providers" and not body.cloud_consent:
        raise ValueError("Enable cloud sharing for this request before sending audio.")
    if (
        body.demo in ("acoustics", "articulation", "calibration", "providers", "feedback")
        and not body.clip_id
    ):
        raise ValueError("Choose a recording first.")
    if body.demo == "feedback":
        if not body.axis_id:
            raise ValueError("Fit and select a direction first.")
        store.read("axes", body.axis_id)
    for ident in set(
        body.clip_ids
        + body.negative_ids
        + body.positive_ids
        + body.reference_ids
        + body.anchor_ids
        + ([body.clip_id] if body.clip_id else [])
    ):
        store.read("clips", ident)
    clip = store.read("clips", body.clip_id) if body.clip_id else {}
    region = {"start": clip.get("region_start", 0), "end": clip.get("region_end")}

    def run(progress):
        if body.demo == "acoustics":
            return acoustics.analyze(body.clip_id, ceiling=body.ceiling, **region)
        if body.demo in ("embeddings", "omni"):
            return embeddings.compare(body.clip_ids, body.model, body.layer, progress)
        if body.demo == "articulation":
            return articulation.analyze(body.clip_id, **region, progress=progress)
        if body.demo == "directions":
            axis = directions.train(
                body.negative_ids,
                body.positive_ids,
                body.name,
                body.negative_label,
                body.positive_label,
                body.model,
                body.layer,
                body.group_by,
                progress,
            )
            return {
                "kind": "trained-axis",
                **{k: v for k, v in axis.items() if k not in ("mean", "scale", "coef")},
            }
        if body.demo == "calibration":
            return calibration.compare(
                body.clip_id, body.reference_ids, body.anchor_ids, body.model, body.layer, progress
            )
        if body.demo == "providers":
            return providers.assess(body.clip_id, body.provider, body.reference, body.language, progress)
        prediction = directions.predict(body.clip_id, body.axis_id, progress)
        prediction["condition"] = body.condition
        prediction["kind"] = "feedback"
        return prediction

    return jobs.submit(run)


@app.get("/api/jobs/{ident}")
def job(ident: str):
    if ident not in jobs.JOBS:
        raise HTTPException(404, "Job not found (the server may have restarted)")
    return jobs.JOBS[ident]


app.mount("/", StaticFiles(directory=ROOT / "speechlab" / "static", html=True), name="app")
