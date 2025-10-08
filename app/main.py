from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from loguru import logger
from typing import Optional
import os

# RQ/Redis imports
import redis
from rq import Queue, Retry

# Import models and service
from .models import UploadResponse, AnalysisRequest, AnalysisStatusResponse
from .file_upload_service import save_audio_file 
from .analysis_worker import perform_analysis_job # Import the worker function

# --- Configuration & Initialization ---
app = FastAPI()

# 1. Initialize Redis connection and RQ Queue
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
logger.info(f"Connecting to Redis at: {REDIS_URL}")
try:
    redis_conn = redis.from_url(REDIS_URL)
    # Ping to check connection immediately
    redis_conn.ping() 
except Exception as e:
    logger.error(f"FATAL: Could not connect to Redis: {e}")
    # You might want to let the app start but fail on /submit
    redis_conn = None 

# 2. Dependency Injection for the Queue
def get_analysis_queue():
    """Provides the RQ queue instance."""
    if redis_conn is None:
        raise HTTPException(status_code=503, detail="Analysis service is unavailable (Redis connection failed).")
    # We use the 'default' queue, which the worker listens to
    return Queue('default', connection=redis_conn)

# --- 1. Root Endpoint (Health Check) ---
@app.get("/")
def read_root():
    """Simple root endpoint to verify the service is running."""
    return {"status": "ok", "message": "PodiumAI backend is running."}

# --- 2. File Upload Endpoint (Existing) ---
@app.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """Receives an audio file, saves it, and returns a file_id."""
    try:
        result = await save_audio_file(file)
        return result 
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- 3. Submit Analysis Job Endpoint (NEW) ---
@app.post("/api/v1/analysis/submit", response_model=AnalysisStatusResponse)
async def submit_analysis_job(
    request: AnalysisRequest, 
    queue: Queue = Depends(get_analysis_queue)
):
    """Submits a long-running analysis job to the worker queue."""
    
    # 1. Reconstruct the full file path (assuming FileUploadService did this)
    # NOTE: You must ensure this matches the logic in file_upload_service.py
    file_path = os.path.join(os.getenv('UPLOAD_DIR', 'uploads_simple'), f"{request.file_id}.m4a")
    
    # 2. Enqueue the job with the required parameters
    try:
        job = queue.enqueue(
            perform_analysis_job, 
            request.file_id, 
            file_path, 
            request.transcript,
            job_id=f"analysis-{request.file_id}", # Consistent job ID
            retry=Retry(max=3) # Retry up to 3 times on failure
        )
        logger.info(f"Analysis job submitted for {request.file_id}. Job ID: {job.id}")
        
        return AnalysisStatusResponse(
            job_id=job.id,
            status=job.get_status(), # 'queued'
            result=None
        )
        
    except Exception as e:
        logger.error(f"Failed to enqueue analysis job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to submit analysis job to the queue: {str(e)}"
        )

# --- 4. Check Job Status Endpoint (NEW) ---
@app.get("/api/v1/analysis/status/{job_id}", response_model=AnalysisStatusResponse)
def get_analysis_status(
    job_id: str, 
    queue: Queue = Depends(get_analysis_queue)
):
    """Checks the status of an analysis job and returns the result if finished."""
    
    job = queue.fetch_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
        
    status = job.get_status()

    if status == 'finished':
        # Result is guaranteed to be the dictionary returned by analysis_worker.py
        result_data = job.result
        
        # FastAPI will automatically validate and convert result_data 
        # into the AnalysisResultResponse Pydantic model
        return AnalysisStatusResponse(
            job_id=job.id,
            status=status,
            result=result_data, # This is the full analysis data
            error=None
        )
    
    elif status == 'failed':
        return AnalysisStatusResponse(
            job_id=job.id,
            status=status,
            error=str(job.exc_info) if job.exc_info else "Job failed for an unknown reason."
        )

    # For 'queued', 'started', or 'deferred'
    return AnalysisStatusResponse(
        job_id=job.id,
        status=status,
        result=None,
        error=None
    )
