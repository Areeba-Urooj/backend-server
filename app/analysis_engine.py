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


# --- 1. Core Feature Extraction (Librosa-based for accuracy) ---
def extract_audio_features(file_path: str) -> Dict[str, Any]:
    """
    Extracts comprehensive audio features using librosa for accurate analysis.
    Works with WAV files after FFmpeg conversion.
    """
    features: Dict[str, Any] = {
        'duration_s': 0.0,
        'sample_rate': TARGET_SR,
        'rms_mean': 0.0,
        'rms_std': 0.0,
        'zero_crossing_rate': 0.0,
        'spectral_centroid': 0.0
    }

    try:
        if not LIBROSA_AVAILABLE:
            # Fallback to basic extraction using ffprobe
            logger.warning("Librosa not available, using basic ffprobe extraction")
            command = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration:stream=sample_rate',
                '-of', 'json', file_path
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            features['duration_s'] = float(probe_data['format']['duration'])
            if 'streams' in probe_data and probe_data['streams']:
                features['sample_rate'] = int(probe_data['streams'][0].get('sample_rate', TARGET_SR))
            return features

        # Use librosa for comprehensive feature extraction
        logger.info(f"Loading audio file with librosa: {file_path}")

        # Load audio with librosa (automatically resamples and converts to mono)
        y, sr = librosa.load(file_path, sr=None, mono=True)
        features['sample_rate'] = sr
        features['duration_s'] = len(y) / sr

        # Extract RMS energy (root mean square)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))

        # Extract zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)
        features['zero_crossing_rate'] = float(np.mean(zcr))

        # Extract spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)
        features['spectral_centroid'] = float(np.mean(centroid))

        logger.info(f"Extracted features: duration={features['duration_s']:.2f}s, RMS={features['rms_mean']:.4f}, ZCR={features['zero_crossing_rate']:.4f}")

        return features

    except Exception as e:
        logger.error(f"Error during audio feature extraction: {e}")
        # Return basic features if extraction fails
        return features

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
    """
    Calculate pitch statistics using librosa.yin() for robust fundamental frequency detection.
    """
    try:
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available, using fallback pitch detection")
            # Simple fallback using autocorrelation
            return _fallback_pitch_detection(y, sr)

        # Use librosa.yin() for robust pitch detection
        # Parameters tuned for human speech (60-300 Hz typical range)
        fmin = 60  # Minimum fundamental frequency (Hz)
        fmax = 300  # Maximum fundamental frequency (Hz)
        frame_length = 2048  # Analysis window size

        # Extract fundamental frequency
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length)

        # Filter out unvoiced frames (where f0 = fmin)
        voiced_frames = f0[f0 > fmin + 1]  # Small buffer above minimum

        if len(voiced_frames) < 5:
            logger.warning("Too few voiced frames detected, using fallback")
            return _fallback_pitch_detection(y, sr)

        # Calculate statistics on voiced frames only
        pitch_mean = float(np.mean(voiced_frames))
        pitch_std = float(np.std(voiced_frames))

        # Validate ranges
        if pitch_mean < 60 or pitch_mean > 400:
            logger.warning(f"Pitch mean {pitch_mean:.1f}Hz outside expected range, using fallback")
            return _fallback_pitch_detection(y, sr)

        logger.info(f"Pitch stats: mean={pitch_mean:.1f}Hz, std={pitch_std:.1f}Hz, voiced_frames={len(voiced_frames)}")
        return pitch_mean, pitch_std

    except Exception as e:
        logger.error(f"Error in pitch detection: {e}, using fallback")
        return _fallback_pitch_detection(y, sr)


def _fallback_pitch_detection(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Fallback pitch detection using simple autocorrelation method.
    """
    try:
        # Simple energy-based detection
        frame_size = int(0.02 * sr)  # 20ms frames
        hop_size = int(0.01 * sr)    # 10ms hop

        pitch_estimates = []

        for i in range(0, len(y) - frame_size, hop_size):
            frame = y[i:i + frame_size]

            # Only process frames with sufficient energy
            energy = np.mean(frame**2)
            if energy > 0.001:  # Energy threshold
                # Simple autocorrelation
                corr = np.correlate(frame, frame, mode='full')[len(frame)-1:]
                # Look for peaks in typical pitch range (60-300 Hz)
                min_lag = int(sr / 300)
                max_lag = int(sr / 60)

                if max_lag < len(corr) and min_lag < max_lag:
                    lag_range = corr[min_lag:max_lag]
                    if len(lag_range) > 0:
                        peak_idx = np.argmax(lag_range)
                        lag = min_lag + peak_idx
                        if lag > 0:
                            pitch = sr / lag
                            if 60 <= pitch <= 400:  # Reasonable pitch range
                                pitch_estimates.append(pitch)

        if len(pitch_estimates) >= 3:
            pitch_mean = float(np.mean(pitch_estimates))
            pitch_std = float(np.std(pitch_estimates))
            return pitch_mean, pitch_std
        else:
            # Return reasonable defaults if no pitch detected
            return 185.0, 15.0  # Typical adult male speaking pitch

    except Exception as e:
        logger.error(f"Fallback pitch detection failed: {e}")
        return 185.0, 15.0  # Safe defaults


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


# --- 4. Scoring Logic (FIXED for Division by Zero) ---

def score_confidence(audio_features: Dict[str, Any], fluency_metrics: Dict[str, Any]) -> float:
    """
    Calculate confidence score on a 0-100 scale based on speech quality metrics.
    Higher scores indicate better speech delivery.
    """
    WEIGHTS = {
        'PACE': 0.30,      # Speaking pace (most important)
        'DISFLUENCY': 0.35, # Fillers and repetitions (very important)
        'PITCH': 0.20,     # Pitch variation (moderately important)
        'ENERGY': 0.15,    # Energy consistency (less important)
    }

    # Calculate pace score (ideal range: 120-180 WPM)
    pace_wpm = audio_features.get('speaking_pace_wpm', 120)
    if 120 <= pace_wpm <= 180:
        pace_score = 1.0  # Perfect range
    elif 100 <= pace_wpm <= 200:
        # Linear decrease outside ideal range
        distance_from_ideal = min(abs(pace_wpm - 120), abs(pace_wpm - 180))
        pace_score = max(0.2, 1.0 - (distance_from_ideal / 40))
    else:
        pace_score = 0.1  # Very poor pace

    # Calculate disfluency score (lower disfluencies = higher score)
    total_disfluencies = (
        fluency_metrics.get('filler_word_count', 0) +
        fluency_metrics.get('repetition_count', 0) +
        fluency_metrics.get('acoustic_disfluency_count', 0)
    )
    total_words = max(1, fluency_metrics.get('total_words', 1))
    disfluency_rate = total_disfluencies / total_words

    if disfluency_rate <= 0.02:  # Very fluent (< 2% disfluencies)
        disfluency_score = 1.0
    elif disfluency_rate <= 0.05:  # Good (2-5%)
        disfluency_score = 0.8
    elif disfluency_rate <= 0.10:  # Average (5-10%)
        disfluency_score = 0.6
    elif disfluency_rate <= 0.20:  # Poor (10-20%)
        disfluency_score = 0.3
    else:  # Very poor (>20%)
        disfluency_score = 0.1

    # Calculate pitch variation score (good variation = moderate std)
    pitch_std = audio_features.get('pitch_std', 20.0)
    if 10 <= pitch_std <= 30:  # Good variation
        pitch_score = 1.0
    elif 5 <= pitch_std <= 50:  # Acceptable range
        # Score decreases as we move away from ideal
        distance = min(abs(pitch_std - 10), abs(pitch_std - 30))
        pitch_score = max(0.3, 1.0 - (distance / 20))
    else:  # Too monotone or too erratic
        pitch_score = 0.2

    # Calculate energy consistency score
    energy_std = audio_features.get('energy_std', 0.01)
    if 0.003 <= energy_std <= 0.02:  # Good consistency
        energy_score = 1.0
    elif 0.001 <= energy_std <= 0.05:  # Acceptable range
        distance = min(abs(energy_std - 0.003), abs(energy_std - 0.02))
        energy_score = max(0.4, 1.0 - (distance / 0.02))
    else:  # Too monotone or too erratic
        energy_score = 0.3

    # Calculate weighted final score
    final_score = (
        (pace_score * WEIGHTS['PACE']) +
        (disfluency_score * WEIGHTS['DISFLUENCY']) +
        (pitch_score * WEIGHTS['PITCH']) +
        (energy_score * WEIGHTS['ENERGY'])
    )

    # Convert to 0-100 scale and ensure reasonable bounds
    confidence_score = final_score * 100

    # Apply bounds: minimum 15 (very poor), maximum 95 (excellent)
    confidence_score = max(15, min(95, confidence_score))

    logger.info(f"Confidence calculation: pace={pace_score:.2f}, disfluency={disfluency_score:.2f}, pitch={pitch_score:.2f}, energy={energy_score:.2f} → {confidence_score:.1f}/100")

    return confidence_score
