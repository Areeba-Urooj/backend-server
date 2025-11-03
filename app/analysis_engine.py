# analysis_engine.py

# ... (Existing imports: numpy, soundfile, json, subprocess, typing, etc.)
# Ensure subprocess and json are imported here!
import subprocess 
import json
import soundfile as sf
import numpy as np
from typing import Dict, Any, List, Tuple, NamedTuple, Optional
# ... (other necessary imports like sklearn, keras, etc.)

# --- Constants ---
TARGET_SR = 16000
MAX_DURATION_SECONDS = 120 # Example limit

# Define NamedTuple for Acoustic Disfluencies
class DisfluencyResult(NamedTuple):
    type: str # 'stutter', 'block'
    start_time_s: float
    duration_s: float

# --- Feature Extraction Helper (FFmpeg/FFprobe safe) ---
def extract_audio_features(file_path: str, max_duration: Optional[int] = None) -> Dict[str, Any]:
    """
    Safely extracts duration using ffprobe for any format, 
    and uses soundfile for detailed features ONLY if it's a WAV file.
    
    In the worker, this is used for duration check on the raw M4A, 
    and then for feature extraction on the converted WAV.
    """
    features: Dict[str, Any] = {'duration_s': 0.0}

    # 1. Safely extract duration using ffprobe (works for M4A, MP3, etc.)
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration:stream=sample_rate',
            '-of', 'json',
            file_path
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        
        duration = float(probe_data['format']['duration'])
        features['duration_s'] = duration
        
        # If this function is called with the M4A file (before conversion), 
        # we only need the duration. We skip soundfile.read.
        if not file_path.lower().endswith('.wav'):
             return features

    except subprocess.CalledProcessError as e:
        # If ffprobe fails (e.g., if the file is truly corrupt or not found)
        raise RuntimeError(f"FFprobe duration extraction failed: {e.stderr}")
    except Exception as e:
        # If JSON loading or another issue occurs
        raise RuntimeError(f"Error during audio feature extraction (FFprobe): {e}")

    # 2. Extract detailed features using soundfile (Only runs if file_path is expected to be WAV)
    try:
        y, sr = sf.read(file_path, dtype='float32', always_2d=False)
        # ... (rest of feature calculation logic using y and sr, if needed)
        # Since the worker re-calculates many features using numpy on y/sr, 
        # we can keep this function minimal, primarily focused on the duration check 
        # for the worker's M4A step.
        
        return features # Return what was extracted (primarily duration for the M4A step)

    except sf.LibsndfileError as e:
        # This should only happen if the file is a WAV but corrupt, or if
        # the worker mistakenly called this with M4A/non-WAV after the FFprobe check.
        raise RuntimeError(f"Audio feature extraction failed (Soundfile): {e}")

# --- (Rest of analysis_engine.py functions must be present) ---

# Example/Placeholder functions that must exist for the worker to run:

# Example of a textual analysis function (must return list and count)
def detect_fillers(transcript: str) -> Tuple[List[str], int]:
    fillers = ["um", "uh", "like", "so"]
    found_fillers = [word for word in transcript.lower().split() if word in fillers]
    return found_fillers, len(found_fillers)

def detect_repetitions(transcript: str) -> Tuple[List[str], int]:
    # Placeholder for actual repetition logic
    return [], 0 

# Placeholder for ML model loading (must return model, scaler, and a placeholder)
def initialize_emotion_model():
    return None, None, None 

# Placeholder for acoustic functions
def classify_emotion_simple(wav_file_path, model, scaler) -> str:
    return "Neutral"

def calculate_pitch_stats(y, sr) -> Tuple[float, float]:
    return 120.0, 10.0

def detect_acoustic_disfluencies(y, sr) -> List[DisfluencyResult]:
    # Placeholder
    return [DisfluencyResult(type='block', start_time_s=1.5, duration_s=0.3)]

def score_confidence(audio_features: Dict[str, Any], fluency_metrics: Dict[str, Any]) -> float:
    # Placeholder
    return 0.85
