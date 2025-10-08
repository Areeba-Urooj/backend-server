from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# --- Upload Response Model (Existing) ---
class UploadResponse(BaseModel):
    """Defines the expected JSON structure after a successful file upload."""
    status: str
    file_id: str
    file_name: str
    file_size: int
    content_type: str
    upload_time: Optional[str] = None
    
# --- Analysis Request Model (NEW) ---
class AnalysisRequest(BaseModel):
    """Data received from the Flutter app to start an analysis job."""
    file_id: str
    transcript: str
    user_id: str # For potential future use/tracking

# --- Nested Feature Model (NEW) ---
class AudioFeatures(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    rms_mean: float
    rms_std: float
    pitch_mean: float
    pitch_std: float
    pitch_min: float
    pitch_max: float
    speaking_pace: float
    silence_ratio: float
    zcr_mean: float
    pitch_variation_score: float

# --- Analysis Result Model (NEW) ---
class AnalysisResultResponse(BaseModel):
    """The final structure returned by the analysis worker."""
    file_id: str
    file_name: str
    duration_seconds: float
    audio_features: AudioFeatures
    filler_word_analysis: Dict[str, Any] # Contains count, ratio, and map of filler words
    repetition_count: int
    long_pause_count: int
    total_words: int
    confidence_score: float
    emotion: str
    pitch_variation_score: float
    recommendations: List[str]
    analyzed_at: Optional[str] = None

# --- Analysis Status Model (NEW) ---
class AnalysisStatusResponse(BaseModel):
    """Response structure for the status check endpoint."""
    job_id: str
    status: str # e.g., 'queued', 'started', 'finished', 'failed'
    result: Optional[AnalysisResultResponse] = None
    error: Optional[str] = None
# returns a dict that will be serialized as is for simplicity.
# You can add other models here as your backend grows (e.g., AnalysisJob, AnalysisResult)

