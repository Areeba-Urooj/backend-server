# app/main.py

import os
import sys
import logging
from uuid import uuid4
from typing import Dict, Any, Optional, List 

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import boto3
from botocore.exceptions import ClientError

import redis
from rq import Queue
from rq.job import Job
from pydantic import BaseModel, Field, ValidationError 

# Ensure app path is included
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Pydantic Models for Data Structures ---
class TextMarker(BaseModel):
    type: str
    word: str 
    start_char_index: int
    end_char_index: int

class AnalysisResult(BaseModel):
    confidence_score: float
    speaking_pace: int
    filler_word_count: int
    repetition_count: int
    long_pause_count: int  # Change from float to int
    silence_ratio: float
    avg_amplitude: float
    pitch_mean: float
    pitch_std: float
    emotion: str
    energy_std: float
    recommendations: List[str]
    transcript: str
    total_words: Optional[int] = None
    duration_seconds: Optional[float] = None
    transcript_markers: List[Dict[str, Any]] = Field(default_factory=list)


class AnalysisStatusResponse(BaseModel):
    job_id: str
    status: str
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    result: Optional[AnalysisResult] = None 
    error: Optional[str] = None

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

# --- Pydantic Models for Response ---
class SubmissionResponse(BaseModel):
    file_id: str
    job_id: str
    message: str

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
    file_uuid = str(uuid4())
    s3_key = f"uploads/{file_uuid}{file_extension}"

    try:
        logger.info(f"[API] ⬆️ Starting S3 upload to key: {s3_key}")
        
        s3_client.upload_fileobj(
            Fileobj=file.file,
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={
                'ContentType': file.content_type
            }
        )
        
        logger.info(f"[API] ✅ S3 upload complete for S3 Key: {s3_key}")

        return JSONResponse(content={
            "file_id": file_uuid,
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
    s3_key: str = Form(..., description="The S3 Key of the uploaded file."),
    transcript: str = Form(..., description="The full transcription of the audio file."),
    user_id: str = Form(..., description="The ID of the authenticated user.")
):
    """Queues an audio analysis job using s3_key and transcript."""
    if not queue:
        raise HTTPException(status_code=503, detail="RQ/Redis service is unavailable.")

    file_id = os.path.splitext(os.path.basename(s3_key))[0]

    try:
        job = queue.enqueue(
            'app.analysis_worker.perform_analysis_job', 
            kwargs={
                'file_id': file_id,
                's3_key': s3_key,
                'transcript': transcript,
                'user_id': user_id,
            },
            job_timeout='30m', 
            serializer='json' 
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

@app.get("/api/v1/analysis/status/{job_id}", response_model=AnalysisStatusResponse)
def get_analysis_status(job_id: str):
    """Retrieves the status and result of an RQ job."""
    if not redis_conn:
        raise HTTPException(status_code=503, detail="RQ/Redis service is unavailable.")
        
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        status = job.get_status()
        
        logger.info(f"[API] Job {job_id} status: {status}")
        
        response_data = AnalysisStatusResponse(
            job_id=job_id,
            status=status,
            enqueued_at=job.enqueued_at.isoformat() if job.enqueued_at else None,
            started_at=job.started_at.isoformat() if job.started_at else None,
        )

        if status == 'finished':
            response_data.ended_at = job.ended_at.isoformat() if job.ended_at else None
            job_result = job.result
            
            if job_result and isinstance(job_result, dict):
                
                try:
                    # Attempt to create the Pydantic model with the job result
                    analysis_result = AnalysisResult(**job_result)
                    response_data.result = analysis_result
                
                except ValidationError as e:
                    error_message = f"{e.__class__.__name__} for AnalysisResult:\n{e.errors()}"
                    logger.error(f"[API] ❌ FAILED to map job result to AnalysisResult model for job {job_id}:\n{error_message}", exc_info=True)
                    
                    response_data.status = 'failed'
                    response_data.error = f"Result processing error (API): {str(e)}. Worker result keys/types are likely mismatched."
                
                except Exception as e:
                    logger.error(f"[API] ❌ General error processing job result for job {job_id}: {e}", exc_info=True)
                    response_data.status = 'failed'
                    response_data.error = f"General result processing error (API): {str(e)}"
            
            elif status == 'finished' and not job_result:
                logger.error(f"[API] Job {job_id} finished, but result was None.")
                response_data.status = 'failed'
                response_data.error = "Job finished successfully, but worker returned no result data."

        # Explicitly handle 'failed' status from the worker
        if status == 'failed' and job.exc_info:
            response_data.error = job.exc_info
            logger.error(f"[API] Full exception info for worker job {job_id}:\n{job.exc_info}")
            
        return response_data
        
    except redis.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail="Could not connect to Redis.")
    except Exception as e:
        logger.error(f"[API] Error fetching job {job_id}: {e}", exc_info=True)
        if 'No such job' in str(e) or 'NoneType' in str(e): 
             raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found.")
        else:
             raise HTTPException(status_code=500, detail=f"Internal error fetching job status: {str(e)}")
