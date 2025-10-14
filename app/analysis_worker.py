# app/analysis_worker.py

import os
import logging
from typing import Dict, Any, Optional
import time

import boto3
from botocore.exceptions import ClientError
from redis import Redis
from rq import Worker
import numpy as np
import librosa
# This import is now safe because the startup command is 'python -m app.analysis_worker'
from app.analysis_engine import (
    detect_fillers,
    detect_repetitions,
    score_confidence,
    classify_emotion,
    initialize_emotion_model
)

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION', 'eu-north-1') 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [WORKER] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- S3 Client Initialization ---
def get_s3_client():
    """Initializes and returns the S3 client with the correct region."""
    if not S3_BUCKET_NAME:
        logger.error("[WORKER] ❌ S3_BUCKET_NAME environment variable is not set.")
        raise ValueError("S3_BUCKET_NAME is not configured.")
    return boto3.client('s3', region_name=AWS_REGION)

try:
    s3_client = get_s3_client()
    logger.info(f"[WORKER] ✅ S3 Client initialized for bucket: {S3_BUCKET_NAME} in region: {AWS_REGION}")
except ValueError as e:
    logger.error(f"[WORKER] ❌ Configuration Error: {e}")
    raise

# Initialize emotion classification model
# emotion_scaler is returned but not used directly in this worker logic, 
# relying on classify_emotion to handle any necessary scaling internally or 
# assuming simple unscaled features for now.
try:
    emotion_model, _, model_created = initialize_emotion_model()
    if model_created:
        logger.info("[WORKER] ✅ Created new emotion classification model in memory.")
    else:
        logger.info("[WORKER] ✅ Loaded existing emotion classification model from disk.")
except Exception as e:
    logger.error(f"[WORKER] ❌ CRITICAL: Error initializing emotion model: {e}", exc_info=True)
    # Re-raise or let the worker fail if model is critical.
    # For now, we continue but use a robust fallback in classify_emotion.

# --- Core Analysis Function (No changes needed) ---
def perform_analysis_job(
    file_id: str, 
    s3_key: str, 
    transcript: str, 
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Worker function to fetch audio, perform analysis, and return the results.
    """
    job_start_time = time.time()
    logger.info(f"🚀 Starting analysis for file_id: {file_id}, s3_key: {s3_key}")
    if user_id:
        logger.info(f"User ID: {user_id}")

    temp_audio_file = f"/tmp/{file_id}_{os.path.basename(s3_key)}"
    duration_seconds = 0
    total_words = len(transcript.split())

    try:
        # 1. Download the file from S3
        logger.info(f"⬇️ Downloading s3://{S3_BUCKET_NAME}/{s3_key} to {temp_audio_file}")
        s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_audio_file)
        logger.info("✅ Download complete.")
        
        # 2. Perform Audio Analysis using librosa (Example Logic)
        y, sr = librosa.load(temp_audio_file, sr=None)
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        
        # Calculate RMS (Root Mean Square Energy)
        rms = librosa.feature.rms(y=y)[0]
        
        # Simple placeholder for silence/pauses (e.g., RMS below a threshold)
        rms_threshold = np.mean(rms) * 0.2
        silence_ratio = np.sum(rms < rms_threshold) / len(rms)
        long_pause_count = int(silence_ratio * 10)
        
        # Simple Placeholder for Pitch (F0)
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=300)
        pitch_valid = pitches[magnitudes > np.quantile(magnitudes, 0.9)]
        pitch_mean = np.mean(pitch_valid) if len(pitch_valid) > 0 else 0
        pitch_std = np.std(pitch_valid) if len(pitch_valid) > 0 else 0
        
        # Speaking Pace (words per minute)
        speaking_pace_wpm = (total_words / max(1, duration_seconds)) * 60

        # Use new analysis engine functions
        filler_word_count = detect_fillers(transcript)
        repetition_count = detect_repetitions(transcript)
        
        # Prepare audio features for confidence scoring
        audio_features = {
            "rms_mean": np.mean(rms),
            "rms_std": np.std(rms),
            "speaking_pace_wpm": speaking_pace_wpm,
            "pitch_std": pitch_std
        }
        
        # Prepare fluency metrics for confidence scoring
        fluency_metrics = {
            "filler_word_count": filler_word_count,
            "repetition_count": repetition_count,
            "total_words": total_words
        }
        
        # Calculate confidence score using new algorithm
        confidence_score = score_confidence(audio_features, fluency_metrics)
        
        # Classify emotion using the ML model
        emotion = classify_emotion(temp_audio_file, emotion_model)
        
        # Generate recommendations based on analysis
        recommendations = []
        if filler_word_count > 0:
            recommendations.append(f"Try to reduce filler words (detected: {filler_word_count}).")
        if repetition_count > 0:
            recommendations.append(f"Work on avoiding repetitions (detected: {repetition_count}).")
        if speaking_pace_wpm < 120:
            recommendations.append("Consider speaking a bit faster to maintain engagement.")
        elif speaking_pace_wpm > 160:
            recommendations.append("Try to slow down your speaking pace for better clarity.")
        if silence_ratio > 0.3:
            recommendations.append("Reduce long pauses for better flow.")
        if confidence_score < 0.7:
            recommendations.append("Work on vocal consistency and energy to improve confidence.")
        
        if not recommendations:
            recommendations = ["Excellent speech clarity and delivery!"]

        # 3. Compile Results to match AnalysisResult Pydantic model
        analysis_result = {
            "confidence_score": confidence_score,
            "speaking_pace": int(round(speaking_pace_wpm)),
            "filler_word_count": filler_word_count,
            "repetition_count": repetition_count,
            "long_pause_count": float(long_pause_count),
            "silence_ratio": round(silence_ratio, 2),
            "avg_amplitude": round(np.mean(rms), 4),
            "pitch_mean": round(pitch_mean, 1),
            "pitch_std": round(pitch_std, 1),
            "emotion": emotion,
            "energy_std": round(np.std(rms), 4),
            "recommendations": recommendations,
            "transcript": transcript,
        }

        logger.info("✅ Analysis complete.")
        
        # 4. Clean up the temporary file
        os.remove(temp_audio_file)
        logger.info(f"🗑️ Cleaned up temp file: {temp_audio_file}")
        
        # 5. Return the full result structure
        return analysis_result

    except ClientError as e:
        logger.error(f"❌ S3 Error during worker processing: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ General Analysis Error: {e}", exc_info=True)
        raise
    finally:
        # Final cleanup attempt
        if os.path.exists(temp_audio_file):
             os.remove(temp_audio_file)

# --- Worker Entrypoint (No changes needed) ---
if __name__ == '__main__':
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    logger.info(f"Starting worker and connecting to Redis at: {redis_url}")
    
    try:
        redis_conn = Redis.from_url(redis_url)
        redis_conn.ping()
        logger.info("Redis connection established.")
        
        # The FIX was applied in the Render start command: 'python -m app.analysis_worker'
        worker = Worker(['default'], connection=redis_conn)
        worker.work()

    except Exception as e:
        logger.error(f"❌ Worker failed to start or connect to Redis: {e}", exc_info=True)
        exit(1)
