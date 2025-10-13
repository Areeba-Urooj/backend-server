from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# --- 1. File Upload Model ---
class UploadResponse(BaseModel):
    """
    Defines the expected JSON structure after a successful file upload.
    The file_id field contains the full S3 Key (path within the bucket).
    """
    status: str
    file_id: str = Field(..., description="The S3 Key (path/filename) of the uploaded object.")
    file_name: str
    file_size: int
    content_type: str
    upload_time: Optional[str] = None

# --- 2. Analysis Submission Model (Input) ---
class AnalysisRequest(BaseModel):
    """
    Defines the expected JSON structure for submitting a job.
    """
    # 💡 UPDATED DESCRIPTION
    file_id: str = Field(..., description="The S3 Key of the audio file returned by the /upload endpoint.")
    transcript: str = Field(..., description="The full, unedited transcript of the audio.")

# --- 3. Analysis Status/Result Model (Output) ---
class AnalysisStatusResponse(BaseModel):
# ... (No change required) ...
    job_id: str
    status: str = Field(..., description="Job status: 'queued', 'started', 'finished', or 'failed'.")
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
