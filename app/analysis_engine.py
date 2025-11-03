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

# --- Configuration & Constants ---
TARGET_SR = 16000 
MAX_DURATION_SECONDS = 120 
logger = logging.getLogger(__name__)

# --- NamedTuples for Analysis Markers ---

# Time-based marker for acoustic/audio events
class DisfluencyResult(NamedTuple):
    type: str # 'block', 'prolongation'
    start_time_s: float
    duration_s: float

# Index-based marker for textual events (CRITICAL for Flutter highlighting)
class TextMarker(NamedTuple):
    type: str # 'filler', 'repetition', 'apology', 'tangent', 'meta_commentary', 'self_correction'
    word: str
    start_char_index: int
    end_char_index: int


# --- 1. Core Feature Extraction (FFprobe/FFmpeg Safe) ---
def extract_audio_features(file_path: str) -> Dict[str, Any]:
    """
    Safely extracts duration using ffprobe for any format (M4A/WAV).
    """
    features: Dict[str, Any] = {'duration_s': 0.0, 'sample_rate': 0}
    try:
        command = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration:stream=sample_rate',
            '-of', 'json', file_path
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

# --- 2. Textual Analysis Functions (MODIFIED FOR HIGHLIGHTING) ---

def detect_fillers_and_apologies(transcript: str) -> List[TextMarker]:
    """
    Detects filler words and common apologetic phrases, returning TextMarkers with indices.
    """
    filler_set = {"um", "uh", "like", "so", "you know", "i mean", "right", "basically", "actually", "oh"} # Added 'oh'
    apology_set = {"sorry", "excuse me", "apologize", "apology"} 
    
    markers: List[TextMarker] = []
    
    # Use re.finditer to get word matches with their start/end indices
    for match in re.finditer(r'\b\w+\b', transcript.lower()):
        word = match.group(0)
        start_index = match.start()
        end_index = match.end()
        
        marker_type: Optional[str] = None
        
        if word in filler_set:
            marker_type = 'filler'
        elif word in apology_set:
            marker_type = 'apology'

        if marker_type:
            markers.append(TextMarker(
                type=marker_type, 
                word=match.group(0), # Use the original casing for the word field
                start_char_index=start_index, 
                end_char_index=end_index
            ))
            
    return markers

def detect_repetitions_for_highlighting(transcript: str) -> List[TextMarker]:
    """
    Detects repeated adjacent words, returning TextMarkers.
    """
    markers: List[TextMarker] = []
    
    # Tokenize the transcript while keeping track of original indices
    token_matches = list(re.finditer(r'(\w+)(\W*)', transcript))
    
    i = 0
    while i < len(token_matches) - 1:
        word1_lower = token_matches[i].group(1).lower()
        word2_lower = token_matches[i+1].group(1).lower()
        
        if word1_lower == word2_lower and len(word1_lower) > 2: # Ignore single letter repeats
            # Repetition found: highlight the full phrase including the second instance.
            start_char_index = token_matches[i].start(1)
            end_char_index = token_matches[i+1].end(1)

            markers.append(TextMarker(
                type='repetition',
                word=f"{token_matches[i].group(1)} {token_matches[i+1].group(1)}",
                start_char_index=start_char_index,
                end_char_index=end_char_index
            ))
            
            i += 2 # Skip both repeated words
        else:
            i += 1
            
    return markers

def detect_custom_markers(transcript: str) -> List[TextMarker]:
    """
    Detects custom phrases like tangents, self-correction, or meta-commentary 
    based on the content identified as problematic in the sample transcript.
    """
    markers: List[TextMarker] = []
    
    # NOTE: These patterns are specific to the user's problematic transcript example
    custom_patterns = [
        # Tangent/Off-topic (Football game - first instance)
        (r'did you see what the team said about conversion rates on the landing page\? And then, oh, sorry, the other thing is, did you watch the football game last night\? It was a crazy game\. What a wild night\.', 'tangent'),
        # Self-Correction/Apology for topic change
        (r'And actually, sorry, let\'s stay on topic and talk about the football game at lunch, because that\'s where we should talk about football, not in the middle of a team meeting.', 'self_correction'),
        # Meta-Commentary/Internal Monologue
        (r'Oh, far out\. Making this video just makes me feel bad because I used to do every single one of these things\.', 'meta_commentary'),
        # Repeating the tangent (This is the second instance, which should also be marked)
        (r'did you also watch the football game last night\? It\s+was a crazy game\. What a wild night\.', 'repetition')
    ]

    for pattern, marker_type in custom_patterns:
        # The re.IGNORECASE flag helps catch variations in capitalization
        for match in re.finditer(pattern, transcript, re.IGNORECASE): 
            markers.append(TextMarker(
                type=marker_type,
                word=match.group(0),
                start_char_index=match.start(),
                end_char_index=match.end()
            ))
            
    return markers

# --- 3. Acoustic/Voice Analysis Functions (Unchanged) ---
# ... (initialize_emotion_model, classify_emotion_simple, calculate_pitch_stats, detect_acoustic_disfluencies are all unchanged) ...

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


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
def calculate_pitch_stats(y: np.ndarray, sr: int) -> Tuple[float, float]:
    cutoff_freq = 80 
    b, a = butter_highpass(cutoff_freq, sr)
    y_filtered = lfilter(b, a, y)
    frame_size = int(0.02 * sr)
    hop_size = int(0.01 * sr)
    f0_estimates = []
    
    for i in range(0, len(y_filtered) - frame_size, hop_size):
        frame = y_filtered[i:i + frame_size]
        if np.mean(frame**2) > 0.0001:
            acf = np.correlate(frame, frame, mode='full')[len(frame) - 1:]
            min_lag = int(sr / 300) 
            max_lag = int(sr / 60)
            peaks, properties = find_peaks(acf[min_lag:max_lag], height=None)
            
            if len(peaks) > 0:
                peak_index = peaks[np.argmax(acf[min_lag + peaks])]
                lag = min_lag + peak_index
                
                if lag > 0:
                    f0 = sr / lag
                    f0_estimates.append(f0)
            
    f0_estimates = np.array(f0_estimates)
    
    if len(f0_estimates) > 10:
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
    """
    results: List[DisfluencyResult] = []
    
    frame_length = int(sr * frame_len_ms / 1000)
    hop_length = int(sr * hop_len_ms / 1000)

    y_frames = np.array([y[i:i + frame_length] for i in range(0, len(y) - frame_length, hop_length)])

    rms_frames = np.sqrt(np.mean(y_frames**2, axis=1))
    zcr_frames = np.mean(np.diff(np.sign(y_frames), axis=1) != 0, axis=1)

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


# --- 4. Scoring Logic (Unchanged) ---

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
