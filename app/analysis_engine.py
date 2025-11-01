# app/analysis_engine.py

import re
import numpy as np
import librosa
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# --- Configuration for Model Paths ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "emotion_scaler.joblib")

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

# --- Emotional Tone Classification ---
def extract_audio_features(audio_path: str) -> np.ndarray:
    """Extract MFCC features from audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=30)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        features = np.concatenate([
            mfccs_scaled,
            [np.mean(spectral_centroids)],
            [np.mean(spectral_rolloff)],
            [np.mean(zero_crossing_rate)]
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}", file=sys.stderr)
        return np.zeros(16)

def create_emotion_model() -> Tuple[RandomForestClassifier, StandardScaler]:
    """Create and train emotion classification model with synthetic data."""
    np.random.seed(42)
    n_samples = 300
    
    features = []
    labels = []
    
    emotions = ['Neutral', 'High Energy', 'Stress']
    for i, emotion in enumerate(emotions):
        n_class_samples = n_samples // len(emotions)
        
        for _ in range(n_class_samples):
            mfccs = np.random.normal(0, 1, 13)
            
            if emotion == 'Neutral':
                spectral_centroid = np.random.normal(2000, 200)
                spectral_rolloff = np.random.normal(3000, 300)
                zcr = np.random.normal(0.05, 0.01)
            elif emotion == 'High Energy':
                spectral_centroid = np.random.normal(3000, 300)
                spectral_rolloff = np.random.normal(4000, 400)
                zcr = np.random.normal(0.08, 0.02)
            else:
                spectral_centroid = np.random.normal(2500, 400)
                spectral_rolloff = np.random.normal(3500, 500)
                zcr = np.random.normal(0.12, 0.03)
            
            feature_vector = np.concatenate([mfccs, [spectral_centroid], [spectral_rolloff], [zcr]])
            features.append(feature_vector)
            labels.append(emotion)
    
    X = np.array(features)
    y = np.array(labels)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def classify_emotion(audio_path: str, model: RandomForestClassifier, scaler: StandardScaler) -> str:
    """
    Classify the emotional tone of an audio file.
    
    CRITICAL FIX: Now accepts both model AND scaler to properly scale features.
    """
    try:
        # Extract raw features
        features = extract_audio_features(audio_path)
        features = features.reshape(1, -1)
        
        # ✅ FIX: Scale the features using the same scaler used during training
        features_scaled = scaler.transform(features)
        
        # Make prediction on scaled features
        prediction = model.predict(features_scaled)[0]
        
        return prediction
        
    except Exception as e:
        print(f"Error classifying emotion for {audio_path}: {e}", file=sys.stderr)
        return "Neutral"  # Changed to capitalized for consistency

# --- Model Initialization ---
def initialize_emotion_model():
    """Initialize or load the emotion classification model."""
    
    try:
        # Try to load existing model from deployment
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Loaded existing emotion classification model from deployment.")
            return model, scaler, False
    except Exception as e:
        print(f"Error loading existing model. Creating new one: {e}", file=sys.stderr)
    
    # Create new model if loading failed
    print("Creating new emotion classification model in memory.")
    model, scaler = create_emotion_model()
    
    return model, scaler, True
