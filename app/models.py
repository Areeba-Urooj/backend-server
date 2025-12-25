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
    # Core metrics
    confidence_score: float
    speaking_pace: int  # WPM
    total_words: int  # Word count
    duration_seconds: float  # Duration

    # Fluency metrics
    filler_word_count: int  # Filler count
    repetition_count: int  # Repetition count
    apology_count: Optional[int] = 0

    # Acoustic metrics
    long_pause_count: int  # Pause count
    silence_ratio: float  # 0.0-1.0

    # Audio features
    avg_amplitude: float
    pitch_mean: float
    pitch_std: float
    energy_std: float
    emotion: str

    # Content
    recommendations: List[str]
    transcript: str

    # 🔥 CRITICAL: Add this field for transcript highlighting
    transcript_markers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Text markers for highlighting (filler, repetition, apology, etc.)"
    )

# --- 4. Analysis Status/Result Model (Output) ---
class AnalysisStatusResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="Job status: 'queued', 'started', 'finished', or 'failed'.")
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
