# app/main.py

import os
import sys
import logging
from uuid import uuid4
from typing import Dict, Any, Optional, List 

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

@app.post("/upload/pdf", response_model=UploadResponse)
async def upload_pdf_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    recording_id: str = Form(...)
):
    """Receives a PDF analysis report and uploads it to a specific S3 path."""
    if not s3_client or not S3_BUCKET_NAME:
        raise HTTPException(status_code=503, detail="S3 storage service is unavailable.")
    
    # Specific path requested by the client: pdfs/[userId]/[recordingId]_analysis.pdf
    s3_key = f"pdfs/{user_id}/{recording_id}_analysis.pdf"

    try:
        logger.info(f"[API] ⬆️ Starting PDF S3 upload to key: {s3_key}")
        
        s3_client.upload_fileobj(
            Fileobj=file.file,
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={
                'ContentType': 'application/pdf'
            }
        )
        
        logger.info(f"[API] ✅ PDF S3 upload complete for S3 Key: {s3_key}")

        return JSONResponse(content={
            "file_id": recording_id,
            "s3_key": s3_key,
            "message": "PDF uploaded successfully."
        }, status_code=200)

    except ClientError as e:
        logger.error(f"[API] ❌ S3 Error during PDF upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"S3 PDF Upload Failed: {e.response['Error'].get('Message', 'Unknown S3 error')}"
        )
    except Exception as e:
        logger.error(f"[API] ❌ General Error during PDF upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the PDF upload: {str(e)}"
        )

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
            
        
    except redis.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail="Could not connect to Redis.")
    except Exception as e:
        logger.error(f"[API] Error fetching job {job_id}: {e}", exc_info=True)
        if 'No such job' in str(e) or 'NoneType' in str(e): 
             raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found.")
        else:
             raise HTTPException(status_code=500, detail=f"Internal error fetching job status: {str(e)}")
