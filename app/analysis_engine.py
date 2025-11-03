# app/analysis_engine.py

import re
import numpy as np
import soundfile as sf
import os
import sys
from typing import Dict, Any, Tuple, List, NamedTuple
from sklearn.preprocessing import StandardScaler
# Note: We avoid importing librosa for memory reasons.

# --- Global Constants for Pitch Analysis ---
TARGET_SR = 16000
MAX_DURATION_SECONDS = 90 # 1.5 minutes maximum
FRAME_SIZE_MS = 25 # 25ms frame size for ZCR and energy analysis

# --- New: Structured Acoustic Disfluency Result ---
class DisfluencyResult(NamedTuple):
    type: str # 'Block', 'Prolongation', 'Rapid Repetition'
    start_time_s: float
    duration_s: float

# --- Feature Extraction Helpers (No changes needed) ---

def get_audio_duration(wav_file_path: str) -> float:
    """Safely get the duration of an audio file."""
    try:
        info = sf.info(wav_file_path)
        return info.duration
    except Exception as e:
        print(f"Error getting audio duration: {e}", file=sys.stderr)
        return 0.0

def calculate_pitch_stats(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Placeholder for pitch calculation using basic zero-crossing rate (ZCR) approximation.
    """
    frame_size = int(0.025 * sr) # 25ms frame
    hop_size = int(0.01 * sr) # 10ms hop
    
    if len(audio) < frame_size:
        return 0.0, 0.0 # pitch_mean, pitch_std
        
    zcr_values = []
    
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * frame_size)
        zcr_values.append(zcr)
        
    zcr_array = np.array(zcr_values)
    
    pitch_mean_proxy = np.mean(zcr_array) * 1000 
    pitch_std_proxy = np.std(zcr_array) * 1000
    
    return pitch_mean_proxy, pitch_std_proxy

# --- Core Feature Extraction (No changes needed) ---

def extract_audio_features(wav_file_path: str, total_words: int) -> Dict[str, Any]:
    """
    Loads audio, limits duration, and extracts core features (RMS, Pitch, Pace).
    """
    features: Dict[str, Any] = {}
    
    try:
        y, sr = sf.read(wav_file_path, dtype='float32', always_2d=False)

        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        max_samples = sr * MAX_DURATION_SECONDS
        if len(y) > max_samples:
            y = y[:max_samples]
            print(f"⚠️ Audio trimmed to max {MAX_DURATION_SECONDS} seconds.")

        duration_s = len(y) / sr
        
        # 2. RMS (Root Mean Square) for Volume/Energy
        rms_values = np.sqrt(np.mean(y**2))
        features['rms_mean'] = np.mean(rms_values)
        features['rms_std'] = np.std(y) 

        # 3. Pitch Statistics 
        pitch_mean, pitch_std = calculate_pitch_stats(y, sr)
        features['pitch_mean'] = pitch_mean
        features['pitch_std'] = pitch_std

        # 4. Speaking Pace
        features['duration_s'] = duration_s
        features['speaking_pace_wpm'] = (total_words / duration_s) * 60 if duration_s > 0 else 0
        
        return features

    except Exception as e:
        print(f"FATAL: Error during audio feature extraction: {e}", file=sys.stderr)
        raise RuntimeError(f"Audio feature extraction failed: {e}")

# --- FLUENCY FUNCTIONS (UPDATED TO RETURN LISTS) ---

def detect_fillers(transcript: str) -> Tuple[List[str], int]:
    """Detect filler words in the transcript and return the list of matches and the count."""
    filler_words = [
        'um', 'uh', 'like', 'you know', 'i mean', 'right', 'so',  
        'actually', 'basically', 'literally', 'well', 'you see',
        'hmm', 'ah', 'er', 'okay', 'alright', 'sort of', 'kind of'
    ]
    
    transcript_lower = transcript.lower()
    matches = []
    
    for filler in filler_words:
        # Use word boundaries for accurate matching
        pattern = r'\b' + re.escape(filler) + r'\b'
        matches.extend(re.findall(pattern, transcript_lower))
        
    return matches, len(matches)


def detect_repetitions(transcript: str) -> Tuple[List[str], int]:
    """
    Detect immediate word/phrase repetitions in the transcript.
    Returns the list of the repeated phrases/words and the count.
    """
    cleaned_transcript = re.sub(r'[^\w\s]', ' ', transcript)
    cleaned_transcript = re.sub(r'\s+', ' ', cleaned_transcript).strip()
    words = cleaned_transcript.lower().split()
    
    if len(words) < 2:
        return [], 0
    
    repetition_list = []
    
    # 1. Check for immediate word repetitions (e.g., "I I went")
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            repetition_list.append(words[i]) # Store the repeated word itself

    # 2. Check for two-word phrase repetitions (e.g., "I went home I went home")
    if len(words) >= 4:
        for i in range(len(words) - 3):
            phrase1 = ' '.join(words[i:i+2])
            phrase2 = ' '.join(words[i+2:i+4])
            if phrase1 == phrase2:
                repetition_list.append(phrase1)
    
    return repetition_list, len(repetition_list)


# --- NEW: ACOUSTIC DISFLUENCY DETECTION ---

def detect_acoustic_disfluencies(y: np.ndarray, sr: int) -> List[DisfluencyResult]:
    """
    Detects acoustic signs of disfluency (blocks, prolongations) in the raw audio.
    
    Args:
        y: Audio time series (numpy array).
        sr: Sample rate.
        
    Returns:
        List of DisfluencyResult objects (NamedTuple: type, start_time_s, duration_s)
    """
    results: List[DisfluencyResult] = []
    
    # --- Acoustic Parameters ---
    # Convert milliseconds to samples
    frame_samples = int(FRAME_SIZE_MS * sr / 1000)
    hop_samples = int(frame_samples / 2) # 50% overlap

    # Thresholds (Tuned based on general speech characteristics)
    MIN_BLOCK_DURATION_S = 0.5 # 500ms of silence/near-silence
    ENERGY_SILENCE_THRESHOLD = np.std(np.abs(y)) * 0.1 # Very low energy
    PROLONGATION_FRAME_COUNT = int(0.3 * 1000 / FRAME_SIZE_MS) # ~300ms 
    ZCR_STABLE_THRESHOLD = 0.005 # Low ZCR change indicates stable sound (vowel hold)

    if len(y) < frame_samples:
        return results

    # --- 1. Frame-level Analysis ---
    frame_energies = []
    frame_zcr_means = []
    
    for i in range(0, len(y) - frame_samples, hop_samples):
        frame = y[i:i + frame_samples]
        energy = np.mean(np.abs(frame))
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * frame_samples)
        frame_energies.append(energy)
        frame_zcr_means.append(zcr)

    energy_array = np.array(frame_energies)
    zcr_array = np.array(frame_zcr_means)
    
    # Convert to boolean flags
    is_silent = energy_array < ENERGY_SILENCE_THRESHOLD
    
    # --- 2. Block/Silence Detection (Potential 'Blocks') ---
    in_block = False
    block_start_frame = -1
    
    for i, silent in enumerate(is_silent):
        if silent and not in_block:
            in_block = True
            block_start_frame = i
        elif not silent and in_block:
            block_end_frame = i
            duration_frames = block_end_frame - block_start_frame
            duration_s = duration_frames * hop_samples / sr
            
            if duration_s >= MIN_BLOCK_DURATION_S:
                start_s = block_start_frame * hop_samples / sr
                results.append(DisfluencyResult(
                    type='Block (Pause)', 
                    start_time_s=start_s, 
                    duration_s=duration_s
                ))
            in_block = False

    # --- 3. Simple Prolongation Detection (Stable ZCR during non-silent speech) ---
    # Look for frames where ZCR is stable (vowel is held) for > PROLONGATION_FRAME_COUNT
    for i in range(len(zcr_array) - PROLONGATION_FRAME_COUNT):
        # Check if energy is NOT silent AND ZCR variation is low over the window
        is_speech = np.all(energy_array[i : i + PROLONGATION_FRAME_COUNT] > ENERGY_SILENCE_THRESHOLD)
        zcr_stable = np.std(zcr_array[i : i + PROLONGATION_FRAME_COUNT]) < ZCR_STABLE_THRESHOLD
        
        if is_speech and zcr_stable:
             # Found a potential prolongation
             start_s = i * hop_samples / sr
             duration_s = PROLONGATION_FRAME_COUNT * hop_samples / sr
             results.append(DisfluencyResult(
                 type='Prolongation',
                 start_time_s=start_s,
                 duration_s=duration_s
             ))
             # Skip ahead to avoid detecting the same prolonged segment multiple times
             i += PROLONGATION_FRAME_COUNT - 1
             
    # NOTE: Rapid repetition (stuttering) is the hardest to detect without ML. 
    # For now, we rely on the Block and Prolongation proxies.
             
    return results

# --- Speaking Confidence Score (No changes needed) ---
def score_confidence(audio_features: Dict[str, Any], fluency_metrics: Dict[str, Any]) -> float:
    # ... (Keep existing implementation)
    rms_std = audio_features.get('rms_std', 0)
    rms_mean = audio_features.get('rms_mean', 0)
    speaking_pace_wpm = audio_features.get('speaking_pace_wpm', 0)
    pitch_std = audio_features.get('pitch_std', 0)
    
    filler_count = fluency_metrics.get('filler_word_count', 0)
    repetition_count = fluency_metrics.get('repetition_count', 0)
    total_words = fluency_metrics.get('total_words', 1)
    
    # Pace Consistency Score
    pace_ideal = 140
    pace_deviation = abs(speaking_pace_wpm - pace_ideal) / pace_ideal
    pace_score = max(0, 1 - pace_deviation)
    
    # Vocal Energy Score
    energy_score = min(1, rms_mean / 0.1)
    energy_variation_penalty = min(0.3, rms_std * 10)
    energy_score = max(0, energy_score - energy_variation_penalty)
    
    # Fluency Score
    filler_rate = filler_count / total_words
    repetition_rate = repetition_count / total_words
    fluency_score = max(0, 1 - (filler_rate * 10 + repetition_rate * 15))
    
    # Pitch Stability Score
    pitch_score = max(0, 1 - (pitch_std / 100))
    
    # Weighted combination
    weights = {
        'pace': 0.3,
        'energy': 0.3,
        'fluency': 0.3,
        'pitch': 0.1
    }
    
    confidence_score = (
        weights['pace'] * pace_score +
        weights['energy'] * energy_score +
        weights['fluency'] * fluency_score +
        weights['pitch'] * pitch_score
    )
    
    return round(max(0.0, min(1.0, confidence_score)), 2)

# --- Emotion Model (No changes needed) ---
def initialize_emotion_model():
    """Initialize a simple rule-based emotion classifier."""
    print("Initializing rule-based emotion classifier...")
    model = None
    scaler = StandardScaler()
    
    dummy_data = np.random.randn(100, 16)
    scaler.fit(dummy_data)
    
    return model, scaler, True

def classify_emotion_simple(audio_path: str, model=None, scaler=None) -> str:
    # ... (Keep existing implementation)
    try:
        y, sr = sf.read(audio_path, dtype='float32', always_2d=False)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        max_samples = sr * 10 
        if len(y) > max_samples:
            y = y[:max_samples]
            
        energy = np.mean(np.abs(y))
        energy_std = np.std(np.abs(y))
        
        zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y))
        
        frame_size = int(0.1 * sr) 
        frame_energies = []
        for i in range(0, len(y) - frame_size, frame_size):
            frame = y[i:i + frame_size]
            frame_energies.append(np.mean(np.abs(frame)))
            
        tempo_variation = np.std(frame_energies) if frame_energies else 0
        
        if energy > 0.15 and zero_crossings > 0.08 and tempo_variation > 0.05:
            return "High Energy"
        elif energy < 0.05 or tempo_variation < 0.02:
            return "Neutral"
        elif energy_std > 0.08 and tempo_variation > 0.04:
            return "Stress"
        else:
            return "Neutral"
            
    except Exception as e:
        print(f"Error classifying emotion: {e}", file=sys.stderr)
        return "Neutral"
