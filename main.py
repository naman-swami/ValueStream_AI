import os
import tempfile
import asyncio
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Header, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates
from schemas import VideoCaptionResponse, AgenticWorkflowResponse
from fireworks_client import set_current_api_key
from ingestion import async_audio_pipeline, sync_visual_pipeline, validate_video_processable
from orchestrator import run_agentic_workflow, create_job, get_job_status, update_job_status

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)
logger = logging.getLogger("video_cap_pipeline")

app = FastAPI(
    title="ValueStream AI: Enterprise Multi-Agent Video Captioning Pipeline",
    version="2.0.0"
)
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Health Check Endpoint (Instant Response for Container Evaluation)
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Lightweight health check endpoint for container evaluation and readiness probes."""
    return {"status": "ok", "service": "ValueStream AI"}


# ---------------------------------------------------------------------------
# Core Single Video Processor
# ---------------------------------------------------------------------------
async def process_single_video_agentic(
    file: UploadFile,
    api_key: Optional[str] = None,
    job_id: Optional[str] = None
) -> AgenticWorkflowResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    set_current_api_key(api_key)

    tmp_video_path = None
    tmp_audio_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            content = await file.read()
            tmp_video.write(content)
            tmp_video_path = tmp_video.name

        tmp_audio_path = tmp_video_path.replace(".mp4", ".mp3")

        # True Parallel Ingestion (Ffmpeg + Whisper & OpenCV capped at 4 keyframes / max 720p)
        audio_task = async_audio_pipeline(tmp_video_path, tmp_audio_path)
        visual_task = asyncio.to_thread(sync_visual_pipeline, tmp_video_path, 4, 720)

        (final_transcript, has_audio), (frames, scene_changes, duration) = await asyncio.gather(
            audio_task, visual_task
        )

        validate_video_processable(duration, frames, final_transcript, has_audio)

        workflow_resp = await run_agentic_workflow(
            transcript=final_transcript,
            has_audio=has_audio,
            base64_frames=frames,
            scene_changes=scene_changes,
            duration=duration,
            max_iterations=3,
            passing_score=90,
            job_id=job_id
        )

        return workflow_resp

    except HTTPException as e:
        if job_id:
            update_job_status(job_id, status="ERROR", current_stage="error", error=str(e.detail))
        raise
    except Exception as e:
        msg = f"Failed to process video: {str(e)}"
        if job_id:
            update_job_status(job_id, status="ERROR", current_stage="error", error=msg)
        raise HTTPException(status_code=500, detail=msg)
    finally:
        if tmp_video_path and os.path.exists(tmp_video_path):
            try:
                os.remove(tmp_video_path)
            except OSError as e:
                logger.warning(f"[Cleanup Warning] Could not remove {tmp_video_path}: {e}")
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            try:
                os.remove(tmp_audio_path)
            except OSError as e:
                logger.warning(f"[Cleanup Warning] Could not remove {tmp_audio_path}: {e}")


async def process_single_video(file: UploadFile, api_key: Optional[str] = None) -> VideoCaptionResponse:
    workflow_resp = await process_single_video_agentic(file, api_key=api_key)
    return workflow_resp.captions


# ---------------------------------------------------------------------------
# Background Job Runner for Live Polling UI
# ---------------------------------------------------------------------------
async def _background_job_runner(file_content: bytes, filename: str, api_key: Optional[str], job_id: str):
    class VirtualUploadFile:
        def __init__(self, content: bytes, fname: str):
            self.filename = fname
            self._content = content
        async def read(self) -> bytes:
            return self._content

    vfile = VirtualUploadFile(file_content, filename)
    try:
        await process_single_video_agentic(vfile, api_key=api_key, job_id=job_id)
    except Exception as e:
        logger.error(f"[Background Job {job_id} Error] {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/caption", response_model=VideoCaptionResponse)
async def create_video_caption(
    file: UploadFile = File(...),
    x_fireworks_api_key: Optional[str] = Header(None, alias="X-Fireworks-API-Key"),
    api_key: Optional[str] = Form(None)
):
    return await process_single_video(file, api_key=x_fireworks_api_key or api_key)


@app.post("/api/v1/caption/agentic", response_model=AgenticWorkflowResponse)
async def create_video_caption_agentic(
    file: UploadFile = File(...),
    x_fireworks_api_key: Optional[str] = Header(None, alias="X-Fireworks-API-Key"),
    api_key: Optional[str] = Form(None)
):
    return await process_single_video_agentic(file, api_key=x_fireworks_api_key or api_key)


@app.post("/analyze", response_model=AgenticWorkflowResponse)
async def analyze_video_ui(
    file: UploadFile = File(...),
    x_fireworks_api_key: Optional[str] = Header(None, alias="X-Fireworks-API-Key"),
    api_key: Optional[str] = Form(None)
):
    """Synchronous analyze endpoint returning full AgenticWorkflowResponse telemetry."""
    return await process_single_video_agentic(file, api_key=x_fireworks_api_key or api_key)


@app.post("/analyze_job")
async def analyze_video_job(
    file: UploadFile = File(...),
    x_fireworks_api_key: Optional[str] = Header(None, alias="X-Fireworks-API-Key"),
    api_key: Optional[str] = Form(None)
):
    """Asynchronous job launcher for live UI telemetry polling without fake timeouts."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_content = await file.read()
    job_id = create_job()
    effective_key = x_fireworks_api_key or api_key
    asyncio.create_task(_background_job_runner(file_content, file.filename, effective_key, job_id))

    return {"job_id": job_id, "status": "RUNNING"}


@app.get("/job_status/{job_id}")
async def get_job_status_endpoint(job_id: str):
    """Returns real-time execution stage and perf_counter timing metrics for job_id."""
    status_data = get_job_status(job_id)
    if not status_data:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JSONResponse(content=status_data)


@app.post("/api/v1/caption/batch", response_model=List[VideoCaptionResponse])
async def create_video_captions_batch(
    files: List[UploadFile] = File(...),
    x_fireworks_api_key: Optional[str] = Header(None, alias="X-Fireworks-API-Key"),
    api_key: Optional[str] = Form(None)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    tasks = [process_single_video(file, api_key=x_fireworks_api_key or api_key) for file in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_results = []
    for r in results:
        if isinstance(r, Exception):
            raise HTTPException(status_code=500, detail=f"Batch processing error: {str(r)}")
        final_results.append(r)

    return final_results


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/.well-known/{path:path}", include_in_schema=False)
async def well_known_catchall(path: str):
    return Response(status_code=204)
