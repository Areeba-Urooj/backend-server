# analysis_engine.py

import numpy as np
import soundfile as sf
import subprocess
import json
import logging
import re
from typing import Dict, Any, List, Tuple, NamedTuple, Optional
from scipy.signal import find_peaks, butter, lfilter
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler 
# NOTE: librosa has been removed

# --- Configuration & Constants ---
TARGET_SR = 16000 
MAX_DURATION_SECONDS = 120 
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
    Returns basic file features. (Identical to previous version)
    """
    features: Dict[str, Any] = {'duration_s': 0.0, 'sample_rate': 0}
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

# --- 2. Textual Analysis Functions (Identical to previous version) ---

def detect_fillers(transcript: str) -> Tuple[List[str], int]:
    filler_set = {"um", "uh", "like", "so", "you know", "i mean", "right", "basically", "actually"}
    words = re.findall(r'\b\w+\b', transcript.lower())
    found_fillers = [word for word in words if word in filler_set]
    return found_fillers, len(found_fillers)

def detect_repetitions(transcript: str) -> Tuple[List[str], int]:
    words = transcript.lower().split()
    repetitions = []
    count = 0
    i = 0
    while i < len(words) - 1:
        if words[i] == words[i+1]:
            repetitions.append(words[i])
            count += 1
            i += 2 
        else:
            i += 1
    return repetitions, count

# --- 3. Acoustic/Voice Analysis Functions (LIBROSA REMOVED) ---

# --- ML Model Placeholder (Identical to previous version) ---
def initialize_emotion_model():
    class MockModel:
        def predict(self, features):
            return np.array([0]) 
    class MockScaler:
        def transform(self, data):
            return data
    return MockModel(), MockScaler(), ["neutral", "calm", "happy", "sad", "angry"] 

def classify_emotion_simple(wav_file_path: str, model, scaler) -> str:
    emotion_classes = ["neutral", "calm", "happy", "sad", "angry"]
    try:
        y, sr = sf.read(wav_file_path)
        energy = np.mean(y**2)
        if energy > 0.005: 
            return "excited"
        return "neutral" 
    except Exception as e:
        logger.warning(f"Emotion classification failed: {e}")
        return "unknown"


def calculate_pitch_stats(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Calculates Mean and Standard Deviation of Pitch (F0) using Autocorrelation (NumPy/SciPy).
    This is a simpler, less robust approach than Librosa's pYIN, but avoids the Librosa dependency.
    """
    # Simple autocorrelation-based pitch estimation (AMDF or Zero-Crossing Rate combined with ACF is best)
    # Using a simple Zero-Crossing Rate (ZCR) proxy for vocal activity and a basic autocorrelation for pitch mean
    
    # 1. High-pass filter (to help with ZCR and autocorrelation)
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    
    cutoff_freq = 80  # Hz, standard for speech
    b, a = butter_highpass(cutoff_freq, sr)
    y_filtered = lfilter(b, a, y)
    
    frame_size = int(0.02 * sr) # 20 ms
    hop_size = int(0.01 * sr)   # 10 ms
    
    f0_estimates = []
    
    for i in range(0, len(y_filtered) - frame_size, hop_size):
        frame = y_filtered[i:i + frame_size]
        
        # Simple thresholding to detect voiced segments (energy)
        if np.mean(frame**2) > 0.0001: 
            # Autocorrelation (ACF) - Simple but effective for F0
            acf = np.correlate(frame, frame, mode='full')[len(frame) - 1:]
            
            # Find the first prominent peak (period)
            # We look for periods corresponding to frequencies between 60Hz and 300Hz
            min_lag = int(sr / 300) 
            max_lag = int(sr / 60)
            
            # Find peaks in the valid lag range
            peaks, properties = find_peaks(acf[min_lag:max_lag], height=None)
            
            if len(peaks) > 0:
                # The peak index is the lag (period in samples)
                peak_index = peaks[np.argmax(acf[min_lag + peaks])]
                lag = min_lag + peak_index
                
                if lag > 0:
                    f0 = sr / lag
                    f0_estimates.append(f0)
            
    f0_estimates = np.array(f0_estimates)
    
    if len(f0_estimates) > 10:
        # Simple outlier removal (Z-score based)
        f0_filtered = f0_estimates[np.abs(zscore(f0_estimates)) < 2] 
        pitch_mean = np.mean(f0_filtered)
        pitch_std = np.std(f0_filtered)
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
            
    return float(pitch_mean), float(pitch_std)


def detect_acoustic_disfluencies(y: np.ndarray, sr: int, frame_len_ms: int = 25, hop_len_ms: int = 10) -> List[DisfluencyResult]:
    """
    Detects basic acoustic disfluencies (blocks/prolongations) using Energy and ZCR (NumPy/SciPy).
    Replaces Librosa-based detection.
    """
    results: List[DisfluencyResult] = []
    
    frame_length = int(sr * frame_len_ms / 1000)
    hop_length = int(sr * hop_len_ms / 1000)

    # Simple Framing/Windowing
    y_frames = np.array([y[i:i + frame_length] for i in range(0, len(y) - frame_length, hop_length)])

    # Calculate short-term Energy (RMS) and Zero Crossing Rate (ZCR)
    rms_frames = np.sqrt(np.mean(y_frames**2, axis=1))
    # Zero-crossing rate: count sign changes / frame_length
    zcr_frames = np.mean(np.diff(np.sign(y_frames), axis=1) != 0, axis=1)

    # Heuristic Thresholds
    energy_threshold = np.mean(rms_frames) * 0.1
    zcr_threshold = np.mean(zcr_frames) * 0.5 
    
    min_event_frames = int(0.3 / (hop_len_ms / 1000))
    
    is_event = False
    event_start_frame = 0
    
    for i in range(len(rms_frames)):
        
        is_block_candidate = (rms_frames[i] < energy_threshold)
        is_prolongation_candidate = (rms_frames[i] > energy_threshold * 2) and (zcr_frames[i] < zcr_threshold)

        if is_block_candidate or is_prolongation_candidate:
            if not is_event:
                is_event = True
                event_start_frame = i
        else:
            if is_event:
                event_duration_frames = i - event_start_frame
                
                if event_duration_frames >= min_event_frames:
                    
                    start_time_s = event_start_frame * (hop_len_ms / 1000)
                    duration_s = event_duration_frames * (hop_len_ms / 1000)
                    
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


# --- 4. Scoring Logic (Identical to previous version) ---

def score_confidence(audio_features: Dict[str, Any], fluency_metrics: Dict[str, Any]) -> float:
    
    WEIGHTS = {
        'WPM_IDEAL': 0.35, 
        'DISFLUENCY_COUNT': 0.40, 
        'PITCH_STABILITY': 0.15, 
        'ENERGY_STD': 0.10, 
    }

    pace_wpm = audio_features.get('speaking_pace_wpm', 0)
    ideal_pace = 150.0
    pace_deviation = min(0.5, abs(pace_wpm - ideal_pace) / ideal_pace) 
    pace_score = max(0.0, 1.0 - (pace_deviation * 2)) 

    total_disfluencies = (
        fluency_metrics.get('filler_word_count', 0) + 
        fluency_metrics.get('repetition_count', 0) +
        fluency_metrics.get('acoustic_disfluency_count', 0)
    )
    total_words = fluency_metrics.get('total_words', 1)
    max_rate = 0.05 
    disfluency_rate = total_disfluencies / total_words
    disfluency_score = max(0.0, 1.0 - (disfluency_rate / max_rate))

    pitch_std = audio_features.get('pitch_std', 50.0)
    max_pitch_std = 40.0 
    pitch_score = max(0.0, 1.0 - (pitch_std / max_pitch_std))
    
    energy_std = audio_features.get('energy_std', 0.0)
    ideal_energy_std = 0.005 
    energy_deviation = min(0.01, abs(energy_std - ideal_energy_std))
    energy_score = max(0.0, 1.0 - (energy_deviation * 100)) 

    final_score = (
        (pace_score * WEIGHTS['WPM_IDEAL']) +
        (disfluency_score * WEIGHTS['DISFLUENCY_COUNT']) +
        (pitch_score * WEIGHTS['PITCH_STABILITY']) +
        (energy_score * WEIGHTS['ENERGY_STD'])
    )

    return min(1.0, max(0.0, final_score))
