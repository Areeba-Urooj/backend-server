# app/analysis_engine.py

import re
import numpy as np
import soundfile as sf
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
import os
import sys

# --- Filler Word Detection ---
def detect_fillers(transcript: str) -> int:
    """Detect filler words in the transcript."""
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

# --- Repetition/Stutter Detection ---
def detect_repetitions(transcript: str) -> int:
    """Detect repetitions and stutters in the transcript."""
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

# --- Speaking Confidence Score ---
def score_confidence(audio_features: Dict[str, Any], fluency_metrics: Dict[str, Any]) -> float:
    """Calculate speaking confidence score."""
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

# --- Pre-trained Emotion Model ---
def initialize_emotion_model():
    """
    Initialize a simple rule-based emotion classifier.
    Uses audio features to classify emotions without needing a trained model.
    """
    print("Initializing rule-based emotion classifier...")
    
    # We'll use a simple rule-based approach instead of ML model
    # This avoids needing to download/train a model
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
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Limit duration
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
