import os
import uuid
import logging
import re
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

# REMOVED: librosa, numpy, scipy, signal imports (now in analysis_worker.py)

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse # New Import for 202

# --- New RQ/Redis Imports ---
import redis
from rq import Queue
from analysis_worker import perform_analysis_job # Import the heavy logic function

# --- Configuration ---
UPLOAD_DIR = "uploads_simple"
ALLOWED_AUDIO_TYPES = [
    ".wav", ".mp3", ".m4a", ".aac"
]
# FILLER_WORDS list is now ONLY needed in analysis_worker.py

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Redis/RQ Setup ---
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
try:
    redis_conn = redis.from_url(REDIS_URL)
    task_queue = Queue(connection=redis_conn) # RQ Queue
    logger.info("Successfully connected to Redis for RQ.")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}. Check REDIS_URL environment variable.")
    # App will continue, but analysis submission will fail if Redis is dead

# --- FastAPI App ---
app = FastAPI(
    title="PodiumPal Simple Analysis API",
    description="A simplified, asynchronous version of the speech analysis backend.",
    version="1.0.0"
)

# --- Pydantic Schemas (Request/Response Models) ---

class AnalysisRequest(BaseModel):
    file_id: str = Field(..., description="ID of the uploaded audio file to analyze")
    transcript: Optional[str] = Field(None, description="Optional transcript text if already available")

# The following schemas must remain here to validate the response from the worker via the /status endpoint
class AudioFeatures(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    rms_mean: float
    rms_std: float
    pitch_mean: float
    pitch_std: float
    pitch_min: float
    pitch_max: float
    speaking_pace: float
    silence_ratio: float
    zcr_mean: float

class FillerWordAnalysis(BaseModel):
    filler_word_count: int
    filler_words: Dict[str, int]
    filler_word_ratio: float

class SpeechAnalysisResponse(BaseModel):
    file_id: str
    duration_seconds: float
    audio_features: AudioFeatures
    filler_word_analysis: FillerWordAnalysis
    repetition_count: int
    long_pause_count: int
    total_words: int
    confidence_score: float
    emotion: str
    pitch_variation_score: float
    recommendations: List[str]
    analyzed_at: datetime

class UploadResponse(BaseModel):
    file_id: str
    file_name: str
    file_size: int
    content_type: str
    upload_time: datetime
    status: str = "success"

class JobResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[SpeechAnalysisResponse] = None
    detail: Optional[str] = None

# --- Audio Analysis Service Logic ---
# REMOVED: class AudioAnalysisService (moved to analysis_worker.py)

# --- File Upload Service Logic (Kept here for file management) ---

class FileUploadService:
    def __init__(self):
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    async def save_audio_file(self, file: UploadFile) -> Dict[str, Any]:
        logger.info(f"Uploading file: {file.filename}, content type: {file.content_type}, size: {file.size}")
        file_extension = os.path.splitext(file.filename)[1].lower()
        logger.info(f"File extension: {file_extension}")
        if file_extension not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed: {ALLOWED_AUDIO_TYPES}")

        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")

        with open(file_path, "wb") as buffer:
            # Use chunks for large files, though shutil.copyfileobj is fine for smaller ones
            shutil.copyfileobj(file.file, buffer) 
        
        saved_size = os.path.getsize(file_path)
        logger.info(f"Saved file size: {saved_size}")

        return {
            "file_id": file_id,
            "file_name": file.filename,
            "file_size": saved_size,
            "content_type": file.content_type,
            "upload_time": datetime.now()
        }

    def get_file_path(self, file_id: str) -> str:
        for ext in ALLOWED_AUDIO_TYPES:
            file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
            if os.path.exists(file_path):
                return file_path
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")

# --- API Endpoints ---

file_upload_service = FileUploadService()

@app.get("/")
async def read_root():
    return {"message": "PodiumPal Simple Analysis API is running! (Asynchronous Enabled)"}

@app.on_event("startup")
async def startup_event():
    # Clean up old uploads on startup
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

@app.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    try:
        result = await file_upload_service.save_audio_file(file)
        return result
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# NEW ENDPOINTS FOR ASYNCHRONOUS PROCESSING
# ----------------------------------------------------

@app.post("/api/v1/analysis/submit", status_code=status.HTTP_202_ACCEPTED)
async def submit_analysis_job(request: AnalysisRequest):
    """
    Submits the analysis to the background queue and immediately returns a Job ID.
    (Fast Response - HTTP 202 Accepted)
    """
    try:
        # 1. Get the file path (Must succeed, otherwise File Not Found)
        # Note: File is stored on the Web Service's disk for the worker to pick up
        file_path = file_upload_service.get_file_path(request.file_id)
        
        # 2. Enqueue the heavy task
        # IMPORTANT: The worker will delete the file upon completion/failure.
        job = task_queue.enqueue(
            perform_analysis_job, 
            request.file_id, 
            file_path, 
            request.transcript,
            job_timeout='30m' # Allow the job up to 30 minutes to finish
        )
        
        logger.info(f"Analysis job submitted: {job.id}")

        # 3. Return the fast response
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job.id,
                "status": "queued",
                "detail": "Analysis job submitted to background worker."
            }
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error submitting analysis job: {e}", exc_info=True)
        # Check if Redis is the issue
        if 'connection' in str(e).lower() or 'redis' in str(e).lower():
            raise HTTPException(status_code=503, detail="Service Unavailable: Redis connection failed. Check background worker setup.")
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@app.get("/api/v1/analysis/status/{job_id}", response_model=JobResponse)
async def get_analysis_status(job_id: str):
    """
    Checks the status of an ongoing or completed analysis job.
    (Polling Endpoint)
    """
    job = task_queue.fetch_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found.")

    if job.is_finished:
        # The job.result is the dictionary returned by perform_analysis_job
        return {
            "job_id": job_id,
            "status": "complete",
            "result": job.result
        }
    
    if job.is_failed:
        # Check the failure details and log
        error_detail = job.exc_info.splitlines()[-1] if job.exc_info else 'Unknown error'
        logger.error(f"Job {job_id} failed. Reason: {job.exc_info}")
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed in worker: {error_detail}"
        )

    # Job is currently 'queued' or 'started'
    return {
        "job_id": job_id,
        "status": job.get_status(),
        "detail": f"Job status: {job.get_status()}"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# REMOVED: The old synchronous /process endpoint

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
