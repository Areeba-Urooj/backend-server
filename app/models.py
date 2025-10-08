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
    
# You can add other models here as your backend grows (e.g., AnalysisJob, AnalysisResult)
