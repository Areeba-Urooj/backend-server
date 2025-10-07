from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from loguru import logger
import uuid # Needed to generate the job_id
from app.models import UploadResponse, AnalysisRequest, JobSubmissionResponse
from app.file_upload_service import save_audio_file 

# Initialize the FastAPI application
app = FastAPI()

# --- 1. Root Endpoint (Health Check) ---
@app.get("/")
def read_root():
    """
    Simple root endpoint to verify the service is running.
    """
    return {"status": "ok", "message": "PodiumAI backend is running."}

# --- 2. File Upload Endpoint ---
@app.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """
    Receives an audio file, saves it via the service layer, 
    and returns a clean JSON response with the file_id.
    """
    try:
        # Call the service function to save the file and get metadata
        result = await save_audio_file(file)
        
        return result 
        
    except Exception as e:
        # Log the detailed error for debugging
        logger.error(f"Error uploading file: {e}", exc_info=True)
        
        # Raise a clean HTTP 500 error for the client
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred during file upload: {str(e)}"
        )

# ----------------------------------------------------------------------
# --- 3. Analysis Submission Endpoint (FIXES the 404 Error) ---
# ----------------------------------------------------------------------
@app.post("/api/v1/analysis/submit", response_model=JobSubmissionResponse)
async def submit_analysis(request: AnalysisRequest):
    """
    Accepts a file_id and user_id, validates the request, and starts an 
    asynchronous analysis job by returning a unique job_id.
    """
    try:
        # In a real application, you would:
        # 1. Look up the file_id to verify the audio file exists
        # 2. Enqueue a message to your worker process (e.g., Redis Queue)

        # For this test, we immediately generate and return a unique job ID.
        job_id = str(uuid.uuid4())
        
        logger.info(f"Analysis job submitted: file_id={request.file_id}, user_id={request.user_id}, job_id={job_id}")
        
        # Return the structured response defined by JobSubmissionResponse
        return {
            "status": "Job submitted successfully",
            "job_id": job_id
        }
    
    except Exception as e:
        logger.error(f"Error submitting analysis job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to submit analysis job: {str(e)}"
        )

# ----------------------------------------------------------------------
# --- 4. Analysis Status Endpoint (Required for the full workflow) ---
# ----------------------------------------------------------------------
@app.get("/api/v1/analysis/status/{job_id}")
async def get_analysis_status(job_id: str):
    """
    Checks the status of an analysis job using its unique job_id.
    """
    # In a real app, you would check a database or queue status
    # For now, we'll return a placeholder status
    
    # Placeholder Logic:
    # We can fake a 'complete' status if the job_id has an even length
    if len(job_id) % 2 == 0:
        status = "COMPLETE"
        result = {"transcription": "This is a placeholder transcript.", "sentiment": "positive"}
    else:
        status = "PROCESSING"
        result = None

    return {
        "job_id": job_id,
        "status": status,
        "result": result
    }
