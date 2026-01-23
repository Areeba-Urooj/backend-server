# app/main.py

import os
import sys

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
    transcript_markers: List[Dict[str, Any]] = Field(default_factory=list)


class AnalysisStatusResponse(BaseModel):
    ended_at: Optional[str] = None
    result: Optional[AnalysisResult] = None 
    error: Optional[str] = None

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

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


@app.get("/api/v1/pdf/download")
async def download_pdf(s3_key: str):
    """Downloads a PDF from S3 and returns it as a streaming response."""
    if not s3_client or not S3_BUCKET_NAME:
        raise HTTPException(status_code=503, detail="S3 storage service is unavailable.")
    
    try:
        logger.info(f"[API] 📥 Downloading PDF from S3: {s3_key}")
        
        # Get object from S3
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key
        )
        
        # Read the PDF data
        pdf_data = response['Body'].read()
        
        logger.info(f"[API] ✅ PDF downloaded successfully: {s3_key} ({len(pdf_data)} bytes)")
        
        # Return as streaming response
        return StreamingResponse(
            BytesIO(pdf_data),
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{os.path.basename(s3_key)}"',
                'Content-Length': str(len(pdf_data)),
            }
        )
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            logger.error(f"[API] ❌ PDF not found in S3: {s3_key}")
            raise HTTPException(status_code=404, detail=f"PDF not found: {s3_key}")
        else:
            logger.error(f"[API] ❌ S3 Error during PDF download: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"S3 Download Failed: {e.response['Error'].get('Message', 'Unknown S3 error')}"
            )
    except Exception as e:
        logger.error(f"[API] ❌ General Error during PDF download: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while downloading the PDF: {str(e)}"
        )

@app.post("/api/v1/analysis/submit", response_model=SubmissionResponse)
async def submit_analysis_job(
    s3_key: str = Form(..., description="The S3 Key of the uploaded file."),
    transcript: str = Form(..., description="The full transcription of the audio file."),
    user_id: str = Form(..., description="The ID of the authenticated user.")
):
200)

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
    except Exception as e:
        logger.error(f"[API] Error fetching job {job_id}: {e}", exc_info=True)
        if 'No such job' in str(e) or 'NoneType' in str(e): 
             raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found.")
        else:
             raise HTTPException(status_code=500, detail=f"Internal error fetching job status: {str(e)}")

