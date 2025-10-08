from fastapi import FastAPI, File, UploadFile, HTTPException
from loguru import logger
from .models import UploadResponse
from .file_upload_service import save_audio_file 

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
        
        # FastAPI validates the 'result' dictionary against 'UploadResponse'
        # and converts it to clean JSON.
        return result 
        
    except Exception as e:
        # Log the detailed error for debugging
        logger.error(f"Error uploading file: {e}", exc_info=True)
        
        # Raise a clean HTTP 500 error for the client
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred during file upload: {str(e)}"
        )

# --- 3. Placeholder for other endpoints (e.g., Analysis/Status) ---
# Add your remaining endpoints below this line as you build them.
