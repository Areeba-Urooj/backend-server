# app/analysis_engine.py

import re
import numpy as np
import soundfile as sf
import os
import sys
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
# Note: We avoid importing librosa for memory reasons.
# We will use simple scipy/numpy methods for pitch and features.

# --- Global Constants for Pitch Analysis ---
# A common sample rate for speech analysis
TARGET_SR = 16000
# Duration limit to prevent OOM errors on large files
MAX_DURATION_SECONDS = 90 # 1.5 minutes maximum

# --- Feature Extraction Helpers ---

def get_audio_duration(wav_file_path: str) -> float:
    """Safely get the duration of an audio file."""
    try:
        # Use soundfile to read file info without loading full data
        info = sf.info(wav_file_path)
        return info.duration
    except Exception as e:
        print(f"Error getting audio duration: {e}", file=sys.stderr)
        return 0.0

def calculate_pitch_stats(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Placeholder for pitch calculation using basic zero-crossing rate (ZCR) approximation.
    NOTE: Real pitch analysis requires a robust library like librosa/pyaudio.
    This is a low-resource approximation for stability.
    """
    # A simplified, low-resource proxy for pitch variability: ZCR
    # ZCR is the rate at which the signal changes sign. It's related to fundamental frequency (pitch).
    
    # Calculate zero-crossing rate (ZCR) for small, non-overlapping frames
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
    
    # ZCR mean and std are used as proxies for pitch mean/std.
    # Higher ZCR typically means higher frequency/pitch.
    pitch_mean_proxy = np.mean(zcr_array) * 1000 # Scale for a more readable number
    pitch_std_proxy = np.std(zcr_array) * 1000
    
    return pitch_mean_proxy, pitch_std_proxy

# --- Main Feature Extraction Function (Crucial for fixing the worker) ---

def extract_audio_features(wav_file_path: str, total_words: int) -> Dict[str, Any]:
    """
    Loads audio, limits duration, and extracts core features (RMS, Pitch, Pace).
    Uses memory-efficient methods.
    """
    features: Dict[str, Any] = {}
    
    try:
        # 1. Load Audio Data Safely
        # We use sf.read to avoid librosa's high memory footprint.
        y, sr = sf.read(wav_file_path, dtype='float32', always_2d=False)

        # Ensure mono audio
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Limit audio length to prevent memory overload
        max_samples = sr * MAX_DURATION_SECONDS
        if len(y) > max_samples:
            y = y[:max_samples]
            print(f"⚠️ Audio trimmed to max {MAX_DURATION_SECONDS} seconds.")

        duration_s = len(y) / sr
        
        # 2. RMS (Root Mean Square) for Volume/Energy
        rms_values = np.sqrt(np.mean(y**2))
        features['rms_mean'] = np.mean(rms_values)
        features['rms_std'] = np.std(y) # std of amplitude as a proxy for variability

        # 3. Pitch Statistics (Low-Resource Approximation)
        pitch_mean, pitch_std = calculate_pitch_stats(y, sr)
        features['pitch_mean'] = pitch_mean
        features['pitch_std'] = pitch_std

        # 4. Speaking Pace
        features['duration_s'] = duration_s
        features['speaking_pace_wpm'] = (total_words / duration_s) * 60 if duration_s > 0 else 0
        
        return features

    except Exception as e:
        print(f"FATAL: Error during audio feature extraction: {e}", file=sys.stderr)
        # Re-raising the error is crucial for RQ to mark the job as failed
        raise RuntimeError(f"Audio feature extraction failed: {e}")


# --- Filler Word Detection (No changes needed) ---
def detect_fillers(transcript: str) -> int:
    """Detect filler words in the transcript."""
    # ... (Keep existing implementation)
    filler_words = [
        'um', 'uh', 'like', 'you know', 'I mean', 'right', 'so', 
        'actually', 'basically', 'literally', 'well', 'you see',
        'hmm', 'ah', 'er', 'okay', 'alright', 'sort of', 'kind of'
    ]
    
    transcript_lower = transcript.lower()
    filler_count = 0
    for filler in filler_words:
        pattern = r'\b' + re.escape(filler) + r'\b'
        matches = re.findall(pattern, transcript_lower)
        filler_count += len(matches)
    
    return filler_count

# --- Repetition/Stutter Detection (No changes needed) ---
def detect_repetitions(transcript: str) -> int:
    """Detect repetitions and stutters in the transcript."""
    # ... (Keep existing implementation)
    cleaned_transcript = re.sub(r'[^\w\s]', ' ', transcript)
    cleaned_transcript = re.sub(r'\s+', ' ', cleaned_transcript).strip()
    words = cleaned_transcript.lower().split()
    
    if len(words) < 2:
        return 0
    
    repetition_count = 0
    
    # Check for immediate word repetitions
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            repetition_count += 1
    
    # Check for trigram repetitions
    if len(words) >= 3:
        for i in range(len(words) - 2):
            phrase1 = ' '.join(words[i:i+2])
            phrase2 = ' '.join(words[i+1:i+3])
            if phrase1 == phrase2:
                repetition_count += 1
    
    # Check for 4-word phrase repetitions
    if len(words) >= 4:
        for i in range(len(words) - 3):
            phrase1 = ' '.join(words[i:i+2])
            phrase2 = ' '.join(words[i+2:i+4])
            if phrase1 == phrase2:
                repetition_count += 1
    
    return repetition_count

# --- Speaking Confidence Score (No changes needed) ---
def score_confidence(audio_features: Dict[str, Any], fluency_metrics: Dict[str, Any]) -> float:
    """Calculate speaking confidence score."""
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

# --- Pre-trained Emotion Model (No changes needed) ---
def initialize_emotion_model():
    """
    Initialize a simple rule-based emotion classifier.
    """
    print("Initializing rule-based emotion classifier...")
    model = None
    scaler = StandardScaler()
    
    # Fit scaler with dummy data so it's ready to use
    dummy_data = np.random.randn(100, 16)
    scaler.fit(dummy_data)
    
    return model, scaler, True

def classify_emotion_simple(audio_path: str, model=None, scaler=None) -> str:
    """
    Rule-based emotion classification using audio features.
    No ML model needed - uses heuristics based on energy, pitch, and tempo.
    """
    try:
        # Load audio
        # Using sf.read (soundfile) instead of audioread/librosa for stability
        y, sr = sf.read(audio_path, dtype='float32', always_2d=False)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Limit duration (using a smaller limit here for the emotion analysis itself)
        max_samples = sr * 10  # 10 seconds max for speed
        if len(y) > max_samples:
            y = y[:max_samples]
        
        # Calculate key features
        # 1. Energy (amplitude)
        energy = np.mean(np.abs(y))
        energy_std = np.std(np.abs(y))
        
        # 2. Zero crossing rate (indicates pitch/frequency changes)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y))
        
        # 3. Tempo/rhythm variation
        frame_size = int(0.1 * sr)  # 100ms frames
        frame_energies = []
        for i in range(0, len(y) - frame_size, frame_size):
            frame = y[i:i + frame_size]
            frame_energies.append(np.mean(np.abs(frame)))
        
        tempo_variation = np.std(frame_energies) if frame_energies else 0
        
        # Rule-based classification
        if energy > 0.15 and zero_crossings > 0.08 and tempo_variation > 0.05:
            return "High Energy"  # Excited, enthusiastic
        elif energy < 0.05 or tempo_variation < 0.02:
            return "Neutral"  # Calm, monotone
        elif energy_std > 0.08 and tempo_variation > 0.04:
            return "Stress"  # Variable, tense
        else:
            return "Neutral"  # Default
            
    except Exception as e:
        print(f"Error classifying emotion: {e}", file=sys.stderr)
        return "Neutral"
