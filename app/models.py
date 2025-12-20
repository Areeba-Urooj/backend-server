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
    """Complete analysis result with all calculated metrics."""

    # Core Metrics
    confidence_score: float = Field(..., description="Overall confidence score (0-100)")
    speaking_pace: int = Field(..., description="Speaking pace in words per minute (WPM)")
    total_words: int = Field(..., description="Total number of words spoken")
    duration_seconds: float = Field(..., description="Total duration in seconds")

    # Fluency Metrics
    filler_word_count: int = Field(..., description="Number of filler words detected")
    repetition_count: int = Field(..., description="Number of word repetitions detected")
    apology_count: int = Field(default=0, description="Number of apologies detected")

    # Acoustic Metrics
    long_pause_count: int = Field(..., description="Number of long pauses (>0.5s) detected")
    silence_ratio: float = Field(..., description="Ratio of silence to total audio (0.0-1.0)")
    acoustic_disfluency_count: int = Field(default=0, description="Number of acoustic disfluencies")

    # Audio Features
    pitch_mean: float = Field(..., description="Mean pitch in Hz")
    pitch_std: float = Field(..., description="Pitch standard deviation in Hz")
    avg_amplitude: float = Field(default=0.0, description="Average amplitude (RMS)")
    energy_std: float = Field(..., description="Energy standard deviation")

    # Analysis Details
    emotion: str = Field(default="neutral", description="Detected emotion")

    # Text Analysis
    transcript: str = Field(..., description="Full transcription")
    transcript_markers: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Highlighted text markers for UI display"
    )

    # Recommendations
    recommendations: List[str] = Field(..., description="AI-generated improvement recommendations")

# --- 4. Analysis Status/Result Model (Output) ---
class AnalysisStatusResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="Job status: 'queued', 'started', 'finished', or 'failed'.")
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
