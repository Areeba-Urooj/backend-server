from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from loguru import logger
from typing import Optional
import os
import sys
import traceback
import uuid # 💡 NEW: Import uuid for S3 key generation, though file_upload_service might handle it.
import boto3 # 💡 NEW: Import boto3, required for S3 access
from botocore.exceptions import ClientError # 💡 NEW: Import ClientError

# RQ/Redis imports
import redis
from rq import Queue, Retry
from rq.job import Job

# Add the app directory to the Python path to help with imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import models 
from . import file_upload_service # This service will be updated to handle S3 uploads.

# --- Configuration & Initialization ---
app = FastAPI()

# ❌ REMOVED: UPLOAD_DIR is no longer relevant for S3
# UPLOAD_DIR = "uploads" 

# --- AWS S3 Configuration ---
# 💡 NEW: Environment variables for S3 connection
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1') # Default to a common region

def get_s3_client():
    """Initializes and returns the S3 client using environment credentials."""
    if not S3_BUCKET_NAME:
        logger.error("[MAIN] ❌ S3_BUCKET_NAME environment variable is not set.")
        raise HTTPException(status_code=500, detail="S3 configuration missing.")
    try:
        # Boto3 automatically uses AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from env vars
        return boto3.client('s3', region_name=AWS_REGION)
    except Exception as e:
        logger.error(f"[MAIN] ❌ Failed to initialize S3 client: {e}")
        raise HTTPException(status_code=500, detail="AWS S3 service failed to initialize.")


# Initialize Redis connection and RQ Queue (NO CHANGE)
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
# ... (NO CHANGE to / and /health endpoints) ...
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


# --- 2. File Upload Endpoint (Minor Change) ---
# This endpoint now saves to S3 via the service and returns the S3 Key.
@app.post("/upload", response_model=models.UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"[MAIN] 📤 Receiving file upload: {file.filename}")
        # 💡 CHANGE: file_upload_service.save_audio_file must now return the S3 key/ID, not a local file path/ID.
        result = await file_upload_service.save_audio_file(file)
        logger.info(f"[MAIN] ✅ File uploaded to S3 with Key: {result.file_id}")
        return result 
    except ClientError as e: # Catch S3-specific errors
        logger.error(f"[MAIN] ❌ S3 Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"S3 Upload Error: {e.response['Error']['Message']}")
    except Exception as e:
        logger.error(f"[MAIN] ❌ Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- 3. Submit Analysis Job Endpoint (MAJOR Change) ---
# The path variable will now hold the S3 key, and local file checks are removed.
@app.post("/api/v1/analysis/submit", response_model=models.AnalysisStatusResponse)
async def submit_analysis_job(
    request: models.AnalysisRequest,
    queue: Queue = Depends(get_analysis_queue)
):
    # 💡 CHANGE: We assume file_id is the S3 Key.
    logger.info(f"[MAIN] 🚀 Received analysis submission for S3 Key: {request.file_id}")
    s3_key = request.file_id # Rename for clarity, this is the S3 Key/Path
    
    # ❌ REMOVED: No more local file path check. The worker handles S3 download and not-found errors.
    # file_path = os.path.join(UPLOAD_DIR, f"{request.file_id}.m4a")
    
    # if not os.path.exists(file_path):
    #     logger.error(f"[MAIN] ❌ File not found during submission: {file_path}")
    #     raise HTTPException(status_code=404, detail=f"File associated with ID {request.file_id} not found on server.")
    
    # ❌ REMOVED: No more local file size check.
    # file_size = os.path.getsize(file_path)
    # logger.info(f"[MAIN] 📁 File exists: {file_path} ({file_size} bytes)")
    logger.info(f"[MAIN] 📝 Transcript length: {len(request.transcript)} chars")
    
    try:
        # Enqueue the job with the S3 key as the 'file_path' argument
        job = queue.enqueue(
            'app.analysis_worker.perform_analysis_job',
            request.file_id, 
            s3_key, # 💡 CHANGE: Pass the S3 Key instead of the local file_path
            request.transcript,
            job_id=f"analysis-{request.file_id}", 
            retry=Retry(max=3),
            job_timeout='10m',
            result_ttl=3600,  # Keep results for 1 hour
            failure_ttl=3600  # Keep failure info for 1 hour
        )
        
        logger.info(f"[MAIN] ✅ Job enqueued successfully")
        logger.info(f"[MAIN]    Job ID: {job.id}")
        logger.info(f"[MAIN]    Initial status: {job.get_status()}")
        logger.info(f"[MAIN]    Function: {job.func_name}")
        
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

# ... (NO CHANGE to /debug/test-worker-import and /api/v1/analysis/status/{job_id} endpoints) ...
@app.get("/debug/test-worker-import")
def test_worker_import():
    """Test if we can import the worker function"""
    try:
        from . import analysis_worker
        return {
            "status": "success",
            "function_exists": hasattr(analysis_worker, 'perform_analysis_job'),
            "function_path": "app.analysis_worker.perform_analysis_job",
            "module_file": analysis_worker.__file__ if hasattr(analysis_worker, '__file__') else "unknown"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/api/v1/analysis/status/{job_id}", response_model=models.AnalysisStatusResponse)
def get_analysis_status(
    job_id: str, 
    queue: Queue = Depends(get_analysis_queue)
):
    logger.info(f"[MAIN] 🔍 Checking status for job: {job_id}")
    
    try:
        job = queue.fetch_job(job_id)
        
        if not job:
            logger.warning(f"[MAIN] ⚠️  Job not found: {job_id}")
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
