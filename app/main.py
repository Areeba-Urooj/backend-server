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
import analysis_worker 

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

# Pydantic model for submission response
class SubmissionResponse(BaseModel):
    file_id: str
    job_id: str
    message: str

# Pydantic model for upload response
class UploadResponse(BaseModel):
    file_id: str
    s3_key: str
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
    
    file_extension = os.path.splitext(file.filename)[1] or ".m4a"
    file_id = str(uuid4())
    s3_key = f"uploads/{file_id}{file_extension}"

    try:
        logger.info(f"[API] ⬆️ Starting S3 upload to key: {s3_key}")
        file_content = await file.read()
        
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_content,
            ContentType=file.content_type
        )
        logger.info(f"[API] ✅ S3 upload complete for file_id: {file_id}")

        return JSONResponse(content={
            "file_id": file_id,
            "s3_key": s3_key,
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
    file_id: str = Form(..., description="The ID of the file previously uploaded to S3."),
    transcript: str = Form(..., description="The full transcription of the audio file."),
    user_id: str = Form(..., description="The ID of the authenticated user.")
):
    """Queues an audio analysis job using file_id and transcript."""
    if not queue:
        raise HTTPException(status_code=503, detail="RQ/Redis service is unavailable.")

    # In a real app, you would fetch the s3_key from a database using file_id
    # For this architecture, we must assume the s3_key format is derivable or known. 
    # Since the frontend only sends file_id, this endpoint is problematic.
    # We will assume a simple key structure for now.
    s3_key_placeholder = f"uploads/{file_id}.m4a" 
    logger.warning(f"[API] ⚠️ Using placeholder s3_key: {s3_key_placeholder} - This needs to be fetched from a database in a production environment.")


    try:
        # Enqueue the analysis job
        job = queue.enqueue(
            analysis_worker.perform_analysis_job,
            file_id=file_id,
            s3_key=s3_key_placeholder,
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
        # Job.fetch throws an exception if the job ID is not found
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found.")
