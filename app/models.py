# ============================================================================
# FILE 1: app/models.py
# ============================================================================

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
    # ===== AUDIO FEATURES =====
    avg_amplitude: float  # RMS value - 🔥 MUST have value
    pitch_mean: float  # Hz - 🔥 MUST be float, not Optional
    pitch_std: float  # Hz std - 🔥 MUST be float, not Optional
    energy_std: float  # Energy variation
    emotion: str  # neutral, excited, etc.
    
    # ===== CONTENT =====
    recommendations: List[str]  # AI recommendations
    transcript: str  # Full transcription
    
    # ===== HIGHLIGHTING =====
    transcript_markers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Text markers for highlighting (filler, repetition, apology, etc.)"
    )
    
    # ===== OPTIONAL/ADDITIONAL =====
    acoustic_disfluency_count: Optional[int] = None

