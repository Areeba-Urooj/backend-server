from pydantic import BaseModel
from typing import Optional

class UploadResponse(BaseModel):
    """
    Defines the expected JSON structure after a successful file upload.
    This is what the 'curl' command should return.
    """
    status: str
    file_id: str
    file_name: str
    file_size: int
    content_type: str
    upload_time: Optional[str] = None
    

class AnalysisRequest(BaseModel):
    """Defines the request body for submitting a new analysis job."""
    file_id: str
    user_id: str

class JobSubmissionResponse(BaseModel):
    """Defines the response body after successfully submitting a job."""
    status: str
    job_id: str
    
# You may need a model for the status response, but the main.py status endpoint 
# returns a dict that will be serialized as is for simplicity.
# You can add other models here as your backend grows (e.g., AnalysisJob, AnalysisResult)
