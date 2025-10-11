from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from loguru import logger
from typing import Optional
import os

# RQ/Redis imports
import redis
from rq import Queue, Retry

# CRITICAL FIX: Do NOT import analysis_worker module to avoid execution issues
from . import models 
from . import file_upload_service

# --- Configuration & Initialization ---
app = FastAPI()

UPLOAD_DIR = "uploads" 

# Initialize Redis connection and RQ Queue
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
logger.info(f"Connecting to Redis at: {REDIS_URL}")
try:
    redis_conn = redis.from_url(REDIS_URL)
    redis_conn.ping() 
    logger.info("Successfully connected to Redis")
except Exception as e:
    logger.error(f"FATAL: Could not connect to Redis: {e}")
    redis_conn = None 

def get_analysis_queue():
    if redis_conn is None:
        raise HTTPException(status_code=503, detail="Analysis service is unavailable (Redis connection failed).")
    return Queue('default', connection=redis_conn)

# --- 1. Root Endpoint (Health Check) ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "PodiumAI backend is running."}

# --- 2. File Upload Endpoint ---
@app.post("/upload", response_model=models.UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    try:
        result = await file_upload_service.save_audio_file(file)
        return result 
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- 3. Submit Analysis Job Endpoint ---
@app.post("/api/v1/analysis/submit", response_model=models.AnalysisStatusResponse)
async def submit_analysis_job(
    request: models.AnalysisRequest,
    queue: Queue = Depends(get_analysis_queue)
):
    file_path = os.path.join(UPLOAD_DIR, f"{request.file_id}.m4a")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found during submission: {file_path}")
        raise HTTPException(status_code=404, detail=f"File associated with ID {request.file_id} not found on server.")
    
    try:
        # ✅ CRITICAL FIX: Use string-based function reference instead of importing the module
        # RQ will import the function in the worker process, not in the web service
        job = queue.enqueue(
            'app.analysis_worker.perform_analysis_job',  # String reference
            request.file_id, 
            file_path, 
            request.transcript,
            job_id=f"analysis-{request.file_id}", 
            retry=Retry(max=3),
            job_timeout='10m'  # Add timeout to prevent hanging jobs
        )
        logger.info(f"Analysis job submitted for {request.file_id}. Job ID: {job.id}")
        
        return models.AnalysisStatusResponse(
            job_id=job.id,
            status=job.get_status(), 
            result=None
        )
        
    except Exception as e:
        logger.error(f"Failed to enqueue analysis job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to submit analysis job to the queue: {str(e)}"
        )

# --- 4. Check Job Status Endpoint ---
@app.get("/api/v1/analysis/status/{job_id}", response_model=models.AnalysisStatusResponse)
def get_analysis_status(
    job_id: str, 
    queue: Queue = Depends(get_analysis_queue)
):
    job = queue.fetch_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
        
    status = job.get_status()

    if status == 'finished':
        result_data = job.result
        return models.AnalysisStatusResponse(
            job_id=job.id,
            status=status,
            result=result_data, 
            error=None
        )
    
    elif status == 'failed':
        return models.AnalysisStatusResponse(
            job_id=job.id,
            status=status,
            error=str(job.exc_info) if job.exc_info else "Job failed for an unknown reason."
        )

    return models.AnalysisStatusResponse(
        job_id=job.id,
        status=status,
        result=None,
        error=None
    )
