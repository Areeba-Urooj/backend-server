from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from loguru import logger
from typing import Optional
import os
import sys

# RQ/Redis imports
import redis
from rq import Queue, Retry
from rq.job import Job

# Add the app directory to the Python path to help with imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import models 
from . import file_upload_service

# --- Configuration & Initialization ---
app = FastAPI()

UPLOAD_DIR = "uploads" 

# Initialize Redis connection and RQ Queue
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
logger.info(f"[MAIN] Connecting to Redis at: {REDIS_URL}")
try:
    redis_conn = redis.from_url(REDIS_URL)
    redis_conn.ping() 
    logger.info("[MAIN] ✅ Successfully connected to Redis")
except Exception as e:
    logger.error(f"[MAIN] ❌ FATAL: Could not connect to Redis: {e}")
    redis_conn = None 

def get_analysis_queue():
    if redis_conn is None:
        raise HTTPException(status_code=503, detail="Analysis service is unavailable (Redis connection failed).")
    return Queue('default', connection=redis_conn)

# --- 1. Root Endpoint (Health Check) ---
@app.get("/")
def read_root():
    redis_status = "connected" if redis_conn else "disconnected"
    return {
        "status": "ok", 
        "message": "PodiumAI backend is running.",
        "redis": redis_status
    }

# --- Health check endpoint ---
@app.get("/health")
def health_check():
    try:
        if redis_conn:
            redis_conn.ping()
            queue = Queue('default', connection=redis_conn)
            queue_length = len(queue)
            return {
                "status": "healthy", 
                "redis": "connected",
                "queue_length": queue_length
            }
        else:
            return {"status": "unhealthy", "redis": "disconnected"}
    except Exception as e:
        return {"status": "unhealthy", "redis": str(e)}

# --- 2. File Upload Endpoint ---
@app.post("/upload", response_model=models.UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"[MAIN] 📤 Receiving file upload: {file.filename}")
        result = await file_upload_service.save_audio_file(file)
        logger.info(f"[MAIN] ✅ File saved with ID: {result.file_id}")
        return result 
    except Exception as e:
        logger.error(f"[MAIN] ❌ Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- 3. Submit Analysis Job Endpoint ---
@app.post("/api/v1/analysis/submit", response_model=models.AnalysisStatusResponse)
async def submit_analysis_job(
    request: models.AnalysisRequest,
    queue: Queue = Depends(get_analysis_queue)
):
    logger.info(f"[MAIN] 🚀 Received analysis submission for file_id: {request.file_id}")
    file_path = os.path.join(UPLOAD_DIR, f"{request.file_id}.m4a")
    
    if not os.path.exists(file_path):
        logger.error(f"[MAIN] ❌ File not found during submission: {file_path}")
        raise HTTPException(status_code=404, detail=f"File associated with ID {request.file_id} not found on server.")
    
    file_size = os.path.getsize(file_path)
    logger.info(f"[MAIN] 📁 File exists: {file_path} ({file_size} bytes)")
    logger.info(f"[MAIN] 📝 Transcript length: {len(request.transcript)} chars")
    
    try:
        # Enqueue the job with explicit function path
        job = queue.enqueue(
            'app.analysis_worker.perform_analysis_job',
            request.file_id, 
            file_path, 
            request.transcript,
            job_id=f"analysis-{request.file_id}", 
            retry=Retry(max=3),
            job_timeout='10m',
            result_ttl=3600,  # Keep results for 1 hour
            failure_ttl=3600  # Keep failure info for 1 hour
        )
        
        logger.info(f"[MAIN] ✅ Job enqueued successfully")
        logger.info(f"[MAIN]    Job ID: {job.id}")
        logger.info(f"[MAIN]    Initial status: {job.get_status()}")
        logger.info(f"[MAIN]    Function: {job.func_name}")
        
        return models.AnalysisStatusResponse(
            job_id=job.id,
            status=job.get_status(), 
            result=None
        )
        
    except Exception as e:
        logger.error(f"[MAIN] ❌ Failed to enqueue analysis job: {e}", exc_info=True)
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
    logger.info(f"[MAIN] 🔍 Checking status for job: {job_id}")
    
    try:
        job = queue.fetch_job(job_id)
        
        if not job:
            logger.warning(f"[MAIN] ⚠️  Job not found: {job_id}")
            raise HTTPException(status_code=404, detail="Job not found.")
        
        status = job.get_status()
        logger.info(f"[MAIN] 📊 Job {job_id} status: {status}")
        
        if status == 'finished':
            result_data = job.result
            logger.info(f"[MAIN] ✅ Job completed successfully")
            return models.AnalysisStatusResponse(
                job_id=job.id,
                status=status,
                result=result_data, 
                error=None
            )
        
        elif status == 'failed':
            error_message = str(job.exc_info) if job.exc_info else "Job failed for an unknown reason."
            logger.error(f"[MAIN] ❌ Job failed: {error_message}")
            return models.AnalysisStatusResponse(
                job_id=job.id,
                status=status,
                error=error_message
            )
        
        # Job is still queued or started
        return models.AnalysisStatusResponse(
            job_id=job.id,
            status=status,
            result=None,
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MAIN] ❌ Error checking job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error checking job status: {str(e)}")
