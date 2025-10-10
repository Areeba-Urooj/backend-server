from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# --- 1. File Upload Model ---
class UploadResponse(BaseModel):
    """
    Defines the expected JSON structure after a successful file upload.
    """
    status: str
    file_id: str
    file_name: str
    file_size: int
    content_type: str
    upload_time: Optional[str] = None

# --- 2. Analysis Submission Model (Input) ---
class AnalysisRequest(BaseModel):
    """
    Defines the expected JSON structure for submitting a job.
    """
    file_id: str = Field(..., description="The ID returned by the /upload endpoint.")
    transcript: str = Field(..., description="The full, unedited transcript of the audio.")

# --- 3. Analysis Status/Result Model (Output) ---
class AnalysisStatusResponse(BaseModel):
    """
    Defines the expected JSON structure for checking job status or receiving results.
    Note: The 'result' field must match the structure returned by analysis_worker.py.
    """
    job_id: str
    status: str = Field(..., description="Job status: 'queued', 'started', 'finished', or 'failed'.")
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
