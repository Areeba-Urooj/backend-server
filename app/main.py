import os
import sys
import logging
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import boto3
from botocore.exceptions import ClientError

import redis
from rq import Queue
from rq.job import Job
from pydantic import BaseModel

# Add proper path handling for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Configuration ---

# 💡 NEW: AWS S3 Configuration
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION', 'eu-north-1')
S3_URL_PREFIX = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Redis & RQ Setup ---
def get_redis_connection():
    """Initializes and returns the Redis connection."""
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    logger.info(f"[MAIN] Connecting to Redis at: {redis_url}")
    try:
        conn = redis.from_url(redis_url)
        conn.ping()
        logger.info("[MAIN] ✅ Successfully connected to Redis")
        return conn
    except Exception as e:
        logger.error(f"[MAIN] ❌ Failed to connect to Redis: {e}", exc_info=True)
        # 💡 CRITICAL CHANGE: Re-raise the exception to prevent startup if Redis fails
        raise RuntimeError("Failed to connect to Redis for RQ.") from e


redis_conn = get_redis_connection()
queue = Queue('default', connection=redis_conn)

# --- S3 Client Initialization ---
def get_s3_client():
    """Initializes and returns the S3 client."""
    if not S3_BUCKET_NAME:
        logger.error("[MAIN] ❌ S3_BUCKET_NAME environment variable is not set.")
        raise ValueError("S3_BUCKET_NAME is not configured.")
    # Boto3 automatically uses AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from env vars
    return boto3.client('s3', region_name=AWS_REGION)

s3_client = get_s3_client()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Audio Analysis API",
    version="1.0.0",
    description="API for uploading audio and queuing analysis jobs."
)

# CORS configuration to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for the response structure
class AnalysisJobResponse(BaseModel):
    file_id: str
    s3_key: str
    job_id: str
    message: str

# --- API Endpoints ---

@app.get("/")
def health_check():
    """Basic health check."""
    return {"status": "ok", "message": "Audio Analysis API is running."}

@app.post("/upload-and-analyze/", response_model=AnalysisJobResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    transcript: str = ""
):
    """
    Receives an audio file, uploads it to S3, and queues an analysis job.
    """
    logger.info(f"[API] ➡️ Received upload request for file: {file.filename}")

    file_extension = os.path.splitext(file.filename)[1] or ".m4a" # Default to m4a
    file_id = str(uuid4())
    s3_key = f"uploads/{file_id}{file_extension}"

    try:
        # 1. Upload to S3
        logger.info(f"[API] ⬆️ Starting S3 upload to key: {s3_key}")
        file_content = await file.read()
        
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_content,
            ContentType=file.content_type
        )
        logger.info(f"[API] ✅ S3 upload complete for file_id: {file_id}")

        # 2. Enqueue the analysis job
        # IMPORTANT: The worker needs the full path to the analysis function
        from . import analysis_worker 

        job = queue.enqueue(
            analysis_worker.perform_analysis_job,
            file_id=file_id,
            s3_key=s3_key,
            transcript=transcript,
            job_timeout='30m'  # Set timeout for long analysis jobs
        )
        
        logger.info(f"[API] 📝 Job enqueued successfully. Job ID: {job.id}")

        return JSONResponse(content={
            "file_id": file_id,
            "s3_key": s3_key,
            "job_id": job.id,
            "message": "Upload successful. Analysis job queued."
        }, status_code=202)

    except ClientError as e:
        logger.error(f"[API] ❌ S3 Error during upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"S3 Upload Failed: {e.response['Error'].get('Message', 'Unknown S3 error')}"
        )
    except Exception as e:
        logger.error(f"[API] ❌ General Error during upload/enqueue: {e}", exc_info=True)
        # This will catch Redis/RQ errors too
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the request: {str(e)}"
        )

# ... (Job status and result endpoints would typically follow here) ...

@app.get("/job-status/{job_id}")
def get_job_status(job_id: str):
    """Retrieves the status of an RQ job."""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        status = job.get_status()
        
        result_data: Dict[str, Any] = {
            "job_id": job_id,
            "status": status,
            "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None
        }

        if status == 'finished':
            result_data["result"] = job.result
        elif status == 'failed':
            result_data["error"] = job.exc_info
            
        return JSONResponse(content=result_data)
        
    except redis.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail="Could not connect to Redis.")
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found.")
