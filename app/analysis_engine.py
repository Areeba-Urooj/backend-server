# app/analysis_engine.py

import re
import numpy as np
import librosa
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- Filler Word Detection ---
def detect_fillers(transcript: str) -> int:
    """
    Detect filler words in the transcript using a rule-based dictionary approach.
    
    Args:
        transcript: The transcript text to analyze
        
    Returns:
        int: Count of filler words found in the transcript
    """
    # Dictionary of common filler words and phrases
    filler_words = [
        'um', 'uh', 'like', 'you know', 'I mean', 'right', 'so', 
        'actually', 'basically', 'literally', 'well', 'you see',
        'hmm', 'ah', 'er', 'okay', 'alright', 'sort of', 'kind of'
    ]
    
    # Convert transcript to lowercase for case-insensitive matching
    transcript_lower = transcript.lower()
    
    # Count occurrences of each filler word
    filler_count = 0
    for filler in filler_words:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(filler) + r'\b'
        matches = re.findall(pattern, transcript_lower)
        filler_count += len(matches)
    
    return filler_count

# --- Repetition/Stutter Detection ---
def detect_repetitions(transcript: str) -> int:
    """
    Detect repetitions and stutters in the transcript using N-gram analysis.
    
    Args:
        transcript: The transcript text to analyze
        
    Returns:
        int: Count of repetitions found in the transcript
    """
    # Clean the transcript: remove extra whitespace and punctuation
    cleaned_transcript = re.sub(r'[^\w\s]', ' ', transcript)
    cleaned_transcript = re.sub(r'\s+', ' ', cleaned_transcript).strip()
    words = cleaned_transcript.lower().split()
    
    if len(words) < 2:
        return 0
    
    repetition_count = 0
    
    # Check for immediate word repetitions (bigrams)
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            repetition_count += 1
    
    # Check for trigram repetitions (3-word phrases)
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
    """
    Calculate a speaking confidence score based on audio features and fluency metrics.
    
    Args:
        audio_features: Dictionary containing audio features like RMS, pitch, etc.
        fluency_metrics: Dictionary containing fluency metrics like filler count, repetitions
        
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    # Extract features
    rms_std = audio_features.get('rms_std', 0)
    rms_mean = audio_features.get('rms_mean', 0)
    speaking_pace_wpm = audio_features.get('speaking_pace_wpm', 0)
    pitch_std = audio_features.get('pitch_std', 0)
    
    filler_count = fluency_metrics.get('filler_word_count', 0)
    repetition_count = fluency_metrics.get('repetition_count', 0)
    total_words = fluency_metrics.get('total_words', 1)
    
    # Calculate individual component scores (0.0 to 1.0)
    
    # 1. Pace Consistency Score (lower std dev is better)
    # Ideal speaking pace is between 120-160 WPM
    pace_ideal = 140  # middle of ideal range
    pace_deviation = abs(speaking_pace_wpm - pace_ideal) / pace_ideal
    pace_score = max(0, 1 - pace_deviation)
    
    # 2. Vocal Energy Score (higher RMS mean with lower variation is better)
    # Normalize RMS score (assuming typical RMS values are between 0.01 and 0.2)
    energy_score = min(1, rms_mean / 0.1)  # cap at 1.0
    energy_variation_penalty = min(0.3, rms_std * 10)  # penalize high variation
    energy_score = max(0, energy_score - energy_variation_penalty)
    
    # 3. Fluency Score (lower filler/repetition rate is better)
    filler_rate = filler_count / total_words
    repetition_rate = repetition_count / total_words
    fluency_score = max(0, 1 - (filler_rate * 10 + repetition_rate * 15))
    
    # 4. Pitch Stability Score (lower pitch variation is better for confidence)
    pitch_score = max(0, 1 - (pitch_std / 100))  # normalize by typical pitch std
    
    # Weighted combination of scores
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
    
    # Ensure score is between 0.0 and 1.0
    return round(max(0.0, min(1.0, confidence_score)), 2)

# --- Emotional Tone Classification ---
def extract_audio_features(audio_path: str) -> np.ndarray:
    """
    Extract MFCC features from audio file for emotion classification.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        np.ndarray: Extracted features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None, duration=30)  # Limit to 30 seconds for efficiency
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # Add additional features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Combine all features
        features = np.concatenate([
            mfccs_scaled,
            [np.mean(spectral_centroids)],
            [np.mean(spectral_rolloff)],
            [np.mean(zero_crossing_rate)]
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return np.zeros(16)  # Return zero features on error

def create_emotion_model() -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Create and train a simple emotion classification model.
    
    Returns:
        Tuple: (trained model, scaler)
    """
    # Create synthetic training data for demonstration
    # In a real application, you would use a labeled dataset
    np.random.seed(42)
    n_samples = 300
    
    # Generate synthetic features
    features = []
    labels = []
    
    # Generate data for each emotion class
    emotions = ['Neutral', 'High Energy', 'Stress']
    for i, emotion in enumerate(emotions):
        n_class_samples = n_samples // len(emotions)
        
        for _ in range(n_class_samples):
            # Generate synthetic MFCC-like features
            mfccs = np.random.normal(0, 1, 13)
            
            # Add emotion-specific patterns
            if emotion == 'Neutral':
                spectral_centroid = np.random.normal(2000, 200)
                spectral_rolloff = np.random.normal(3000, 300)
                zcr = np.random.normal(0.05, 0.01)
            elif emotion == 'High Energy':
                spectral_centroid = np.random.normal(3000, 300)
                spectral_rolloff = np.random.normal(4000, 400)
                zcr = np.random.normal(0.08, 0.02)
            else:  # Stress
                spectral_centroid = np.random.normal(2500, 400)
                spectral_rolloff = np.random.normal(3500, 500)
                zcr = np.random.normal(0.12, 0.03)
            
            feature_vector = np.concatenate([mfccs, [spectral_centroid], [spectral_rolloff], [zcr]])
            features.append(feature_vector)
            labels.append(emotion)
    
    X = np.array(features)
    y = np.array(labels)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def classify_emotion(audio_path: str, model) -> str:
    """
    Classify the emotional tone of an audio file.
    
    Args:
        audio_path: Path to the audio file
        model: Pre-trained emotion classification model
        
    Returns:
        str: Predicted emotion class
    """
    try:
        # Extract features
        features = extract_audio_features(audio_path)
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return prediction
        
    except Exception as e:
        print(f"Error classifying emotion for {audio_path}: {e}")
        return "Neutral"  # Default fallback

# --- Model Initialization ---
def initialize_emotion_model():
    """
    Initialize or load the emotion classification model.
    Returns the model and a boolean indicating if it was newly created.
    """
    model_path = "emotion_model.joblib"
    scaler_path = "emotion_scaler.joblib"
    
    try:
        # Try to load existing model
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("Loaded existing emotion classification model")
            return model, scaler, False
    except Exception as e:
        print(f"Error loading existing model: {e}")
    
    # Create new model if loading failed or files don't exist
    print("Creating new emotion classification model")
    model, scaler = create_emotion_model()
    
    try:
        # Save the model for future use
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print("Saved new emotion classification model")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    return model, scaler, True
