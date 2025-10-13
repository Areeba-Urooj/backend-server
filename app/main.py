# app/main.py

import os
import sys
import logging
from uuid import uuid4
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import boto3
from botocore.exceptions import ClientError

import redis
from rq import Queue
from rq.job import Job
from pydantic import BaseModel

# Ensure analysis_worker can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_worker  # Worker module is now correctly imported

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION', 'eu-north-1') 

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
        raise RuntimeError("Failed to connect to Redis for RQ.") from e

try:
    redis_conn = get_redis_connection()
    # Use the established connection for the Queue
    queue = Queue('default', connection=redis_conn) 
except RuntimeError:
    # If connection fails, allow FastAPI to start, but job submission will fail
    redis_conn = None
    queue = None

# --- S3 Client Initialization ---
def get_s3_client():
    """Initializes and returns the S3 client."""
    if not S3_BUCKET_NAME:
        logger.error("[MAIN] ❌ S3_BUCKET_NAME environment variable is not set.")
        raise ValueError("S3_BUCKET_NAME is not configured.")
    return boto3.client('s3', region_name=AWS_REGION)

try:
    s3_client = get_s3_client()
    logger.info(f"[MAIN] ✅ S3 Client initialized for bucket: {S3_BUCKET_NAME} in region: {AWS_REGION}")
except ValueError:
    s3_client = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Audio Analysis API",
    version="1.0.0",
    description="API for uploading audio and queuing analysis jobs."
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class SubmissionResponse(BaseModel):
    file_id: str
    job_id: str
    message: str

class UploadResponse(BaseModel):
    file_id: str # This should be the S3 Key (e.g., "uploads/uuid.m4a")
    s3_key: str # Redundant, but kept for compatibility with original logs
    message: str

# --- API Endpoints ---

@app.get("/")
def health_check():
    """Basic health check."""
    return {"status": "ok", "message": "Audio Analysis API is running."}

@app.post("/upload", response_model=UploadResponse)
async def upload_audio_file(
    file: UploadFile = File(...)
):
    """Receives an audio file and uploads it to S3."""
    if not s3_client or not S3_BUCKET_NAME:
        raise HTTPException(status_code=503, detail="S3 storage service is unavailable.")
    
    # Generate S3 key based on file extension
    file_extension = os.path.splitext(file.filename)[1] or ".m4a"
    file_uuid = str(uuid4())
    s3_key = f"uploads/{file_uuid}{file_extension}"

    try:
        logger.info(f"[API] ⬆️ Starting S3 upload to key: {s3_key}")
        
        # Read file content into memory. For large files, use upload_fileobj with a temporary file.
        file_content = await file.read() 
        
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_content,
            ContentType=file.content_type
        )
        logger.info(f"[API] ✅ S3 upload complete for S3 Key: {s3_key}")

        return JSONResponse(content={
            "file_id": file_uuid, # Return UUID for simple reference
            "s3_key": s3_key,     # Return full S3 Key for job submission
            "message": "File uploaded successfully."
        }, status_code=200)

    except ClientError as e:
        logger.error(f"[API] ❌ S3 Error during upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"S3 Upload Failed: {e.response['Error'].get('Message', 'Unknown S3 error')}"
        )
    except Exception as e:
        logger.error(f"[API] ❌ General Error during upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the upload: {str(e)}"
        )

@app.post("/api/v1/analysis/submit", response_model=SubmissionResponse)
async def submit_analysis_job(
    s3_key: str = Form(..., description="The S3 Key (path/filename) of the file returned by the /upload endpoint."),
    transcript: str = Form(..., description="The full transcription of the audio file."),
    user_id: str = Form(..., description="The ID of the authenticated user.")
):
    """Queues an audio analysis job using s3_key and transcript."""
    if not queue:
        raise HTTPException(status_code=503, detail="RQ/Redis service is unavailable.")

    # Extract file_id (UUID) from s3_key for internal tracking/logging
    file_id = os.path.splitext(os.path.basename(s3_key))[0]

    try:
        # Enqueue the analysis job, passing the full S3 key
        job = queue.enqueue(
            analysis_worker.perform_analysis_job,
            file_id=file_id,
            s3_key=s3_key, # Pass the full S3 key
            transcript=transcript,
            user_id=user_id,
            job_timeout='30m'
        )
        
        logger.info(f"[API] 📝 Job enqueued successfully. Job ID: {job.id}")

        return JSONResponse(content={
            "file_id": file_id,
            "job_id": job.id,
            "message": "Analysis job queued."
        }, status_code=200)

    except Exception as e:
        logger.error(f"[API] ❌ Error during job enqueue: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit analysis job: {str(e)}"
        )

@app.get("/api/v1/analysis/status/{job_id}")
def get_job_status(job_id: str):
    """Retrieves the status and result of an RQ job."""
    if not redis_conn:
        raise HTTPException(status_code=503, detail="RQ/Redis service is unavailable.")
        
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        status = job.get_status()
        
        result_data: Dict[str, Any] = {
            "job_id": job_id,
            "status": status,
            "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
        }

        if status == 'finished':
            result_data["result"] = job.result 
            result_data["ended_at"] = job.ended_at.isoformat() if job.ended_at else None
        elif status == 'failed':
            result_data["error"] = job.exc_info
            
        return JSONResponse(content=result_data)
        
    except redis.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail="Could not connect to Redis.")
    except Exception:
        # Job.fetch throws an exception if the job ID is not found
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found.")
