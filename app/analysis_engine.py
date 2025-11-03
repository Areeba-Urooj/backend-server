# analysis_engine.py

import numpy as np
import soundfile as sf
import subprocess
import json
import logging
import re
from typing import Dict, Any, List, Tuple, NamedTuple, Optional
from scipy.signal import find_peaks
import librosa
from sklearn.preprocessing import StandardScaler # Used for confidence scoring/ML pre-processing

# --- Configuration & Constants ---
TARGET_SR = 16000 # Standard sample rate for speech analysis
MAX_DURATION_SECONDS = 120 # Maximum duration for analysis
logger = logging.getLogger(__name__)

# Define NamedTuple for Acoustic Disfluencies
class DisfluencyResult(NamedTuple):
    type: str # 'stutter', 'block', 'prolongation'
    start_time_s: float
    duration_s: float

# --- 1. Core Feature Extraction (FFprobe/FFmpeg Safe) ---

def extract_audio_features(file_path: str) -> Dict[str, Any]:
    """
    Safely extracts duration using ffprobe for any format (M4A/WAV).
    Returns basic file features.
    """
    features: Dict[str, Any] = {'duration_s': 0.0, 'sample_rate': 0}

    # Use ffprobe for safe duration/sample rate check
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
        
        # Try to get sample rate from stream information (more reliable)
        if 'streams' in probe_data and probe_data['streams']:
             sample_rate = int(probe_data['streams'][0].get('sample_rate', TARGET_SR))
             features['sample_rate'] = sample_rate
        else:
             features['sample_rate'] = TARGET_SR
             
        return features

    except subprocess.CalledProcessError as e:
        logger.error(f"FFprobe duration extraction failed: {e.stderr}")
        raise RuntimeError(f"FFprobe duration extraction failed: {e.__class__.__name__}")
    except Exception as e:
        logger.error(f"Error during audio feature extraction (FFprobe): {e}")
        raise RuntimeError(f"Error during audio feature extraction (FFprobe): {e.__class__.__name__}")

# --- 2. Textual Analysis Functions ---

def detect_fillers(transcript: str) -> Tuple[List[str], int]:
    """Identifies and counts common filler words."""
    # Common English filler words
    filler_set = {"um", "uh", "like", "so", "you know", "i mean", "right", "basically", "actually"}
    
    # Simple tokenization
    words = re.findall(r'\b\w+\b', transcript.lower())
    
    found_fillers = [word for word in words if word in filler_set]
    
    return found_fillers, len(found_fillers)

def detect_repetitions(transcript: str) -> Tuple[List[str], int]:
    """Identifies word repetitions (e.g., 'the the dog')."""
    words = transcript.lower().split()
    repetitions = []
    count = 0
    i = 0
    while i < len(words) - 1:
        # Check if the current word is the same as the next word
        if words[i] == words[i+1]:
            repetitions.append(words[i])
            count += 1
            # Skip the next word since it's part of the repetition
            i += 2 
        else:
            i += 1
    
    return repetitions, count

# --- 3. Acoustic/Voice Analysis Functions ---

# --- ML Model Placeholder (since we cannot train a model here) ---
# NOTE: In a real environment, you would load a trained Keras/PyTorch model and a trained Scaler.
def initialize_emotion_model():
    """Initializes a placeholder for the ML emotion model and scaler."""
    # Placeholder for a trained model structure/weights
    class MockModel:
        def predict(self, features):
            # Predicts '0' (Neutral) for simplicity
            return np.array([0]) 

    # Placeholder for a trained Scaler
    class MockScaler:
        def transform(self, data):
            # Does no scaling
            return data

    return MockModel(), MockScaler(), ["neutral", "calm", "happy", "sad", "angry"] # Mock classes


def classify_emotion_simple(wav_file_path: str, model, scaler) -> str:
    """
    Simulates emotion classification. In a real system, this would extract MFCCs, 
    scale them, and use the ML model to predict.
    """
    emotion_classes = ["neutral", "calm", "happy", "sad", "angry"]
    
    # Since we can't fully train/load a model, we'll use a simple heuristic:
    try:
        y, sr = sf.read(wav_file_path)
        # Simplified energy/pitch based emotion
        energy = np.mean(y**2)
        
        if energy > 0.005: # Arbitrary high energy threshold
            return "excited"
        
        # Using a simple deterministic result for stability
        return "neutral" 
    except Exception as e:
        logger.warning(f"Emotion classification failed: {e}")
        return "unknown"


def calculate_pitch_stats(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """Calculates mean and standard deviation of fundamental frequency (F0) using librosa."""
    try:
        # F0 estimation using pYIN (more robust than simple autocorrelation)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), # ~65 Hz
            fmax=librosa.note_to_hz('C7'), # ~2093 Hz
            sr=sr, 
            frame_length=2048, 
            hop_length=512
        )

        # Filter out NaN values (unvoiced frames)
        f0 = f0[~np.isnan(f0)]
        
        if len(f0) > 0:
            pitch_mean = np.mean(f0)
            pitch_std = np.std(f0)
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
            
        return float(pitch_mean), float(pitch_std)
        
    except Exception as e:
        logger.warning(f"Pitch calculation failed: {e}")
        return 0.0, 0.0


def detect_acoustic_disfluencies(y: np.ndarray, sr: int, frame_len_ms: int = 25, hop_len_ms: int = 10) -> List[DisfluencyResult]:
    """
    Detects basic acoustic disfluencies (blocks/prolongations) based on energy and ZCR.
    NOTE: A robust detector requires a trained ML model, this is a heuristic.
    """
    results: List[DisfluencyResult] = []
    
    # Convert ms to samples
    frame_length = int(sr * frame_len_ms / 1000)
    hop_length = int(sr * hop_len_ms / 1000)

    # Calculate short-term energy (RMS) and Zero Crossing Rate (ZCR)
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr_frames = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Heuristic Thresholds
    # Silence/Low Energy Threshold (for 'block')
    energy_threshold = np.mean(rms_frames) * 0.1
    # Prolongation/Sound Repetition (Low ZCR and High Energy over time)
    zcr_threshold = np.mean(zcr_frames) * 0.5 
    
    # Time window for a disfluency event (e.g., > 300ms)
    min_event_frames = int(0.3 / (hop_len_ms / 1000))
    
    is_event = False
    event_start_frame = 0
    
    for i in range(len(rms_frames)):
        
        # Condition for a block (low energy, may or may not be low ZCR)
        is_block_candidate = (rms_frames[i] < energy_threshold)
        
        # Condition for a prolongation (high energy, low ZCR - suggesting a held vowel sound)
        is_prolongation_candidate = (rms_frames[i] > energy_threshold * 2) and (zcr_frames[i] < zcr_threshold)

        if is_block_candidate or is_prolongation_candidate:
            if not is_event:
                # Start of a new event
                is_event = True
                event_start_frame = i
        else:
            if is_event:
                # End of an event
                event_duration_frames = i - event_start_frame
                
                if event_duration_frames >= min_event_frames:
                    
                    start_time_s = event_start_frame * (hop_len_ms / 1000)
                    duration_s = event_duration_frames * (hop_len_ms / 1000)
                    
                    # Determine type based on what was dominant
                    if np.mean(rms_frames[event_start_frame:i]) < energy_threshold:
                         event_type = 'block'
                    else:
                         event_type = 'prolongation'
                         
                    results.append(DisfluencyResult(
                        type=event_type, 
                        start_time_s=round(start_time_s, 2), 
                        duration_s=round(duration_s, 2)
                    ))
                    
                is_event = False
                
    return results


# --- 4. Scoring Logic ---

def score_confidence(audio_features: Dict[str, Any], fluency_metrics: Dict[str, Any]) -> float:
    """
    Calculates a simple confidence score based on key analysis metrics.
    Score is 1.0 (perfect) down to 0.0 (poor).
    """
    # Weights for different factors (Tuned for speech analysis)
    WEIGHTS = {
        'WPM_IDEAL': 0.35, # Speaking Pace
        'DISFLUENCY_COUNT': 0.40, # Filler/Repetition/Acoustic
        'PITCH_STABILITY': 0.15, # Pitch Standard Deviation
        'ENERGY_STD': 0.10, # Volume Variation
    }

    # 1. Pace Score (Target: 140-160 WPM)
    pace_wpm = audio_features.get('speaking_pace_wpm', 0)
    ideal_pace = 150.0
    
    # Calculate deviation from ideal (lower is better, max 0.5 deviation penalized)
    pace_deviation = min(0.5, abs(pace_wpm - ideal_pace) / ideal_pace) 
    pace_score = max(0.0, 1.0 - (pace_deviation * 2)) # Score of 1.0 if deviation is 0, 0.0 if deviation is 0.5 or more

    # 2. Disfluency Score
    total_disfluencies = (
        fluency_metrics.get('filler_word_count', 0) + 
        fluency_metrics.get('repetition_count', 0) +
        fluency_metrics.get('acoustic_disfluency_count', 0)
    )
    total_words = fluency_metrics.get('total_words', 1)
    
    # Max acceptable disfluency rate (e.g., 5% of words)
    max_rate = 0.05 
    disfluency_rate = total_disfluencies / total_words
    
    # Score is 1.0 if rate is 0, 0.0 if rate is max_rate or more
    disfluency_score = max(0.0, 1.0 - (disfluency_rate / max_rate))

    # 3. Pitch Stability Score
    # Standard deviation of pitch (lower is better, max reasonable value is ~50Hz)
    pitch_std = audio_features.get('pitch_std', 50.0)
    max_pitch_std = 40.0 # High standard deviation suggests poor control/erratic pitch
    pitch_score = max(0.0, 1.0 - (pitch_std / max_pitch_std))
    
    # 4. Energy Stability Score (Volume Variation)
    # Standard deviation of RMS/Energy (too high is shouting/too low is whispering)
    energy_std = audio_features.get('energy_std', 0.0)
    ideal_energy_std = 0.005 # A small variation is natural, too much is bad
    
    energy_deviation = min(0.01, abs(energy_std - ideal_energy_std))
    energy_score = max(0.0, 1.0 - (energy_deviation * 100)) # Score of 1.0 if deviation is small

    # 5. Final Weighted Score
    final_score = (
        (pace_score * WEIGHTS['WPM_IDEAL']) +
        (disfluency_score * WEIGHTS['DISFLUENCY_COUNT']) +
        (pitch_score * WEIGHTS['PITCH_STABILITY']) +
        (energy_score * WEIGHTS['ENERGY_STD'])
    )

    return min(1.0, max(0.0, final_score)) # Clamp score between 0.0 and 1.0
