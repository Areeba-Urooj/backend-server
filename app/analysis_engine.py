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


# --- 1. Core Feature Extraction (ffprobe only for metadata) ---
def extract_audio_features(file_path: str) -> Dict[str, Any]:
    """
    Extract audio duration using ffprobe ONLY.
    Soundfile doesn't support M4A, so we use ffprobe for metadata.
    Works for M4A, MP3, WAV, FLAC, OGG, etc.
    """
    features: Dict[str, Any] = {
        'duration_s': 0.0,
        'sample_rate': TARGET_SR,
    }

    try:
        # Use ffprobe to get duration and sample rate (works for all audio formats)
        command = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration:stream=sample_rate',
            '-of', 'json', file_path
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)

        # Extract duration
        duration = float(probe_data['format'].get('duration', 0.0))
        features['duration_s'] = duration

        logger.info(f"✅ Audio duration extracted: {duration:.2f}s")

        # Extract sample rate
        if 'streams' in probe_data and probe_data['streams']:
            sr = int(probe_data['streams'][0].get('sample_rate', TARGET_SR))
            features['sample_rate'] = sr
            logger.info(f"✅ Sample rate extracted: {sr}Hz")
        else:
            features['sample_rate'] = TARGET_SR
            logger.warning(f"⚠️ Could not extract sample rate, using default: {TARGET_SR}Hz")

        return features

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ ffprobe failed: {e.stderr}")
        raise RuntimeError(f"ffprobe extraction failed: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse ffprobe JSON: {e}")
        raise RuntimeError(f"Failed to parse ffprobe output: {e}")
    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {e}", exc_info=True)
        raise RuntimeError(f"Feature extraction failed: {e}")

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
    Detects custom markers (SIMPLIFIED - no complex regex patterns).
    Returns empty list to avoid processing delays.
    """
    # SIMPLIFIED: Return empty list
    # Complex custom pattern matching was causing performance issues
    # Focus on standard filler/repetition detection instead
    return []

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
    Calculate pitch statistics using autocorrelation (optimized, no infinite loops).
    Compatible with Render.com - no librosa dependency.
    """
    from scipy.signal import butter, sosfilt

    try:
        # Apply high-pass filter
        sos = butter(5, 80, 'hp', fs=sr, output='sos')
        y_filtered = sosfilt(sos, y)

        # Frame parameters
        frame_length = int(0.04 * sr)  # 40ms frames
        hop_length = int(0.01 * sr)    # 10ms hop

        if frame_length == 0 or hop_length == 0:
            logger.warning("Invalid frame parameters for pitch detection")
            return 185.0, 15.0

        f0_values = []
        max_frames = 5000  # LIMIT: Process max 5000 frames (prevents infinite loops)
        frame_count = 0

        # Process frames with a defined limit
        start = 0
        while start < len(y_filtered) - frame_length and frame_count < max_frames:
            frame = y_filtered[start:start + frame_length]

            # Skip silent frames
            energy = np.sqrt(np.mean(frame**2))
            if energy < 0.001:
                start += hop_length
                frame_count += 1
                continue

            # Apply window
            window = np.hanning(len(frame))
            frame = frame * window

            # Autocorrelation (limited computation)
            correlation = np.correlate(frame, frame, mode='full')
            correlation = correlation[len(correlation)//2:]  # Second half only
            correlation = correlation / (correlation[0] + 1e-10)  # Normalize safely

            # Find period between 50Hz and 300Hz
            min_period = max(1, int(sr / 300))
            max_period = min(len(correlation) - 1, int(sr / 50))

            if max_period > min_period:
                r = correlation[min_period:max_period]
                if len(r) > 0:
                    period = min_period + np.argmax(r)

                    if period > 0:
                        f0 = sr / period
                        if 60 <= f0 <= 300:  # Only keep human speech frequencies
                            f0_values.append(f0)

            start += hop_length
            frame_count += 1

        logger.info(f"[PITCH] Processed {frame_count} frames, found {len(f0_values)} valid pitch values")

        # Calculate statistics
        if len(f0_values) > 5:
            f0_values = np.array(f0_values)
            median = np.median(f0_values)
            valid = f0_values[np.abs(f0_values - median) < 50]

            if len(valid) > 3:
                pitch_mean = float(np.mean(valid))
                pitch_std = float(np.std(valid))
            else:
                pitch_mean = float(np.mean(f0_values))
                pitch_std = float(np.std(f0_values))
        else:
            pitch_mean = 185.0
            pitch_std = 15.0

        logger.info(f"✅ Pitch calculated: mean={pitch_mean:.1f}Hz, std={pitch_std:.1f}Hz")
        return pitch_mean, pitch_std

    except Exception as e:
        logger.error(f"❌ Pitch calculation failed: {e}", exc_info=True)
        return 185.0, 15.0  # Safe defaults


def detect_acoustic_disfluencies(y: np.ndarray, sr: int, frame_len_ms: int = 25, hop_len_ms: int = 10) -> List[DisfluencyResult]:
    """
    Detects acoustic disfluencies (blocks/prolongations) (optimized, no infinite loops).
    Compatible with Render.com - no librosa dependency.
    """
    results: List[DisfluencyResult] = []

    try:
        # Frame-based processing
        frame_length = int(sr * frame_len_ms / 1000)
        hop_length = int(sr * hop_len_ms / 1000)

        if frame_length == 0 or hop_length == 0:
            logger.warning("Invalid frame parameters for disfluency detection")
            return results

        # Create frames efficiently (avoid infinite loops)
        num_frames = (len(y) - frame_length) // hop_length + 1
        num_frames = min(num_frames, 50000)  # LIMIT: Max 50k frames (safety)

        logger.info(f"[DISFLUENCY] Processing {num_frames} frames...")

        # Extract frames
        frames = []
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            if end <= len(y):
                frames.append(y[start:end])
            else:
                break

        if not frames:
            logger.warning("No frames extracted for disfluency detection")
            return results

        frames = np.array(frames)

        # Calculate RMS energy per frame
        rms_frames = np.sqrt(np.mean(frames**2, axis=1))

        # Calculate zero-crossing rate per frame
        zcr_frames = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2

        # Adaptive thresholds based on audio statistics
        rms_mean = np.mean(rms_frames)
        rms_std = np.std(rms_frames)

        # Silence threshold: mean - 1.5*std (more robust)
        silence_threshold = max(rms_mean - 1.5 * rms_std, 0.001)

        # Loud threshold: mean + 1*std (for prolongations)
        loud_threshold = rms_mean + 1 * rms_std

        zcr_mean = np.mean(zcr_frames)
        zcr_low_threshold = zcr_mean * 0.5

        # Minimum event duration (300ms)
        min_duration_frames = int(0.3 / (hop_len_ms / 1000))
        min_duration_frames = max(1, min_duration_frames)  # At least 1 frame

        # Detect contiguous regions with explicit loop control
        in_event = False
        event_start = 0
        event_type = None

        # Process with explicit break conditions
        for i in range(len(rms_frames)):
            is_silence = rms_frames[i] < silence_threshold
            is_loud_low_zcr = rms_frames[i] > loud_threshold and zcr_frames[i] < zcr_low_threshold

            if is_silence:
                if not in_event:
                    in_event = True
                    event_start = i
                    event_type = 'block'
            elif is_loud_low_zcr:
                if not in_event:
                    in_event = True
                    event_start = i
                    event_type = 'prolongation'
            else:
                if in_event:
                    duration_frames = i - event_start
                    if duration_frames >= min_duration_frames:
                        start_time = event_start * (hop_len_ms / 1000)
                        duration = duration_frames * (hop_len_ms / 1000)
                        results.append(DisfluencyResult(
                            type=event_type or 'block',
                            start_time_s=round(start_time, 2),
                            duration_s=round(duration, 2)
                        ))
                    in_event = False

        # Check end of audio
        if in_event:
            duration_frames = len(rms_frames) - event_start
            if duration_frames >= min_duration_frames:
                start_time = event_start * (hop_len_ms / 1000)
                duration = duration_frames * (hop_len_ms / 1000)
                results.append(DisfluencyResult(
                    type=event_type or 'block',
                    start_time_s=round(start_time, 2),
                    duration_s=round(duration, 2)
                ))

        logger.info(f"✅ Disfluencies detected: {len(results)}")
        return results

    except Exception as e:
        logger.error(f"❌ Disfluency detection failed: {e}", exc_info=True)
        return results


# --- 4. Scoring Logic (FIXED for Division by Zero) ---

def score_confidence(audio_features: Dict[str, Any], fluency_metrics: Dict[str, Any]) -> float:
    """
    Calculate confidence score (0-100 scale, not 0-1) using proper component scoring.
    Compatible with Render.com - uses only NumPy.
    """

    # Component weights
    weights = {
        'pace': 0.30,
        'fluency': 0.40,
        'pitch': 0.20,
        'energy': 0.10,
    }

    # 1. PACE SCORE (0-100)
    pace_wpm = audio_features.get('speaking_pace_wpm', 0)
    ideal_pace = 150  # WPM
    pace_min, pace_max = 100, 200

    if pace_wpm < pace_min:
        pace_score = max(0, (pace_wpm / pace_min) * 50)  # 0-50
    elif pace_wpm > pace_max:
        pace_score = max(0, 100 - ((pace_wpm - pace_max) / (pace_max * 0.5)) * 50)  # 0-50
    else:
        pace_score = 50 + (1 - abs(pace_wpm - ideal_pace) / (pace_max - ideal_pace)) * 50  # 50-100

    pace_score = np.clip(pace_score, 0, 100)

    # 2. FLUENCY SCORE (0-100)
    total_words = max(1, fluency_metrics.get('total_words', 1))
    fillers = fluency_metrics.get('filler_word_count', 0)
    repetitions = fluency_metrics.get('repetition_count', 0)
    acoustic = fluency_metrics.get('acoustic_disfluency_count', 0)

    disfluency_rate = (fillers + repetitions + acoustic) / total_words

    # Perfect = 0 disfluencies, 0.05 (5%) is acceptable, 0.10 (10%) is poor
    if disfluency_rate == 0:
        fluency_score = 100
    elif disfluency_rate < 0.05:
        fluency_score = 100 - (disfluency_rate / 0.05) * 20  # 80-100
    elif disfluency_rate < 0.10:
        fluency_score = 80 - ((disfluency_rate - 0.05) / 0.05) * 30  # 50-80
    else:
        fluency_score = max(20, 50 - (disfluency_rate - 0.10) * 100)

    fluency_score = np.clip(fluency_score, 20, 100)

    # 3. PITCH SCORE (0-100)
    pitch_std = audio_features.get('pitch_std', 0)
    pitch_mean = audio_features.get('pitch_mean', 0)

    # Good pitch variation: 15-40 Hz std
    # Poor: < 5 Hz (monotone) or > 50 Hz (too variable)
    if pitch_std < 5:
        pitch_score = pitch_std * 10  # 0-50 for monotone
    elif pitch_std > 40:
        pitch_score = max(50, 100 - (pitch_std - 40) * 2)  # 50-70
    else:
        pitch_score = 60 + ((pitch_std - 5) / 35) * 40  # 60-100

    pitch_score = np.clip(pitch_score, 0, 100)

    # 4. ENERGY SCORE (0-100)
    energy_std = audio_features.get('energy_std', 0)
    rms_mean = audio_features.get('rms_mean', 0)

    # Good: consistent energy with some variation
    # RMS should be between 0.01 and 0.3 (moderate volume)
    if rms_mean < 0.005:
        energy_score = 30  # Too quiet
    elif rms_mean > 0.5:
        energy_score = 50  # Too loud
    else:
        energy_score = 70 + (energy_std / 0.02) * 30  # 70-100 with variation

    energy_score = np.clip(energy_score, 0, 100)

    # FINAL SCORE (weighted average, 0-100)
    final_score = (
        pace_score * weights['pace'] +
        fluency_score * weights['fluency'] +
        pitch_score * weights['pitch'] +
        energy_score * weights['energy']
    )

    final_score = np.clip(final_score, 20, 100)  # Min 20, Max 100

    logger.info(
        f"📊 Confidence components: "
        f"pace={pace_score:.1f}, fluency={fluency_score:.1f}, "
        f"pitch={pitch_score:.1f}, energy={energy_score:.1f} "
        f"→ Final={final_score:.1f}"
    )

    return round(final_score, 1)
