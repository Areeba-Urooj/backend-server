from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# --- 1. File Upload Model ---
class UploadResponse(BaseModel):
    """
    Defines the expected JSON structure after a successful file upload.
    The file_id field contains the generated UUID, and s3_key contains the full path.
    """
    file_id: str = Field(..., description="The UUID of the uploaded file.")
    s3_key: str = Field(..., description="The S3 Key (path/filename) of the uploaded object. Use this for analysis submission.")
    message: str

# --- 2. Analysis Submission Model (Output) ---
class SubmissionResponse(BaseModel):
    file_id: str
    job_id: str
    message: str

# --- 3. Analysis Result Model ---
class AnalysisResult(BaseModel):
    confidence_score: float  # 0-100 scale (fixed from 0-1)
    speaking_pace: int       # words per minute
    filler_word_count: int
    repetition_count: int
    long_pause_count: int    # Changed from float to int (count of pauses)
    silence_ratio: float
    avg_amplitude: float
    pitch_mean: float
    pitch_std: float
    emotion: str
    energy_std: float
    recommendations: List[str]
    transcript: str
    # Optional fields for additional data
    transcript_markers: Optional[List[Dict]] = None

# --- 4. Analysis Status/Result Model (Output) ---
class AnalysisStatusResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="Job status: 'queued', 'started', 'finished', or 'failed'.")
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
