# app/main.py

import os
import sys
from io import BytesIO

import boto3
import botocore.exceptions

# Ensure app path is included
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Pydantic Models for Data Structures ---

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Redis & RQ Setup ---

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
    """Queues an audio analysis job using s3_key and transcript."""
    if not queue:
        raise HTTPException(status_code=503, detail="RQ/Redis service is unavailable.")

        
    except redis.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail="Could not connect to Redis.")
    except Exception as e:
        logger.error(f"[API] Error fetching job {job_id}: {e}", exc_info=True)
        if 'No such job' in str(e) or 'NoneType' in str(e): 
             raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found.")
        else:
             raise HTTPException(status_code=500, detail=f"Internal error fetching job status: {str(e)}")

