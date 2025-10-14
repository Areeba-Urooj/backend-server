# app/analysis_worker.py

import os
# --- CRITICAL FIX: DISABLE NUMBA JIT TO PREVENT CRASHES IN CONTAINER ENVIRONMENTS ---
os.environ['NUMBA_DISABLE_JIT'] = '1'
# Also enforce native decoding to bypass other potential audio library crashes
os.environ['LIBROSA_USE_NATIVE_MPG123'] = '1'
# -----------------------------------------------------------------------------------

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
emotion_model = None
try:
    emotion_model, _, model_created = initialize_emotion_model()
    if model_created:
        logger.info("[WORKER] ✅ Created new emotion classification model in memory.")
    else:
        logger.info("[WORKER] ✅ Loaded existing emotion classification model from disk.")
except Exception as e:
    logger.error(f"[WORKER] ❌ CRITICAL: Error initializing emotion model: {e}", exc_info=True)
    # The worker can proceed, but emotion classification will default to 'calm'

# --- Core Analysis Function (Modified) ---
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
    y, sr = None, None
    duration_seconds = 0
    total_words = len(transcript.split())
    
    # Default metrics in case of audio processing failure
    default_metrics = {
        "rms_mean": 0.0, "rms_std": 0.0, "silence_ratio": 1.0, 
        "pitch_mean": 0.0, "pitch_std": 0.0, "speaking_pace_wpm": 0.0,
        "emotion": "calm"
    }

    try:
        # 1. Download the file from S3
        logger.info(f"⬇️ Downloading s3://{S3_BUCKET_NAME}/{s3_key} to {temp_audio_file}")
        s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_audio_file)
        logger.info("✅ Download complete.")
        
        # 2. Perform Audio Analysis using librosa
        try:
            # Load the audio file (sr=None to use native sample rate)
            y, sr = librosa.load(temp_audio_file, sr=None)
            duration_seconds = librosa.get_duration(y=y, sr=sr)
            
            if duration_seconds < 0.5:
                 raise Exception("Audio file is too short for meaningful analysis.")

            # Calculate RMS (Root Mean Square Energy)
            # Use max(1e-10, ...) to prevent log(0) issues in further processing, though not strictly needed here
            rms = librosa.feature.rms(y=y)[0]
            
            # Simple placeholder for silence/pauses (e.g., RMS below a threshold)
            # Use max(1e-6, np.mean(rms)) to prevent division by zero for silence threshold
            rms_threshold = np.mean(rms) * 0.2
            silence_ratio = np.sum(rms < rms_threshold) / max(1, len(rms))
            long_pause_count = int(silence_ratio * 10) # Placeholder logic
            
            # Simple Placeholder for Pitch (F0)
            pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=300)
            
            # Robustly select valid pitches
            valid_pitches = pitches[magnitudes > np.quantile(magnitudes, 0.9)]
            
            pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
            pitch_std = np.std(valid_pitches) if len(valid_pitches) > 0 else 0
            
            # Speaking Pace (words per minute)
            speaking_pace_wpm = (total_words / max(1.0, duration_seconds)) * 60

            # Update default metrics with calculated values
            default_metrics.update({
                "rms_mean": np.mean(rms),
                "rms_std": np.std(rms),
                "silence_ratio": silence_ratio,
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "speaking_pace_wpm": speaking_pace_wpm,
            })
            
            # Emotion Classification
            if emotion_model:
                try:
                    default_metrics['emotion'] = classify_emotion(temp_audio_file, emotion_model)
                except Exception as e:
                    logger.warning(f"⚠️ Emotion classification failed: {e}. Defaulting to 'calm'.")
                    default_metrics['emotion'] = "calm"
            
        except Exception as e:
            logger.error(f"❌ Librosa/Audio Processing Failed: {e}", exc_info=True)
            # Keep default metrics, proceed with text analysis if possible
            # The job will not fail yet, but analysis results will be based on defaults/text
            duration_seconds = 1.0 # Ensure duration is not zero for WPM calc if it failed above
            long_pause_count = int(total_words / 5) # Placeholder long pause count
            speaking_pace_wpm = (total_words / max(1.0, duration_seconds)) * 60
            default_metrics['speaking_pace_wpm'] = speaking_pace_wpm


        # Text Analysis (Run regardless of audio success)
        filler_word_count = detect_fillers(transcript)
        repetition_count = detect_repetitions(transcript)
        
        # Prepare audio features for confidence scoring (using calculated or default values)
        audio_features = {
            "rms_mean": default_metrics["rms_mean"],
            "rms_std": default_metrics["rms_std"],
            "speaking_pace_wpm": default_metrics["speaking_pace_wpm"],
            "pitch_std": default_metrics["pitch_std"]
        }
        
        # Prepare fluency metrics for confidence scoring
        fluency_metrics = {
            "filler_word_count": filler_word_count,
            "repetition_count": repetition_count,
            "total_words": total_words
        }
        
        # Calculate confidence score
        confidence_score = score_confidence(audio_features, fluency_metrics)
        
        # Generate recommendations based on analysis
        recommendations = []
        if filler_word_count > 0:
            recommendations.append(f"Try to reduce filler words (detected: {filler_word_count}).")
        if repetition_count > 0:
            recommendations.append(f"Work on avoiding repetitions (detected: {repetition_count}).")
        
        # Use calculated pace or default placeholder
        pace_to_check = default_metrics.get("speaking_pace_wpm", 0)
        if pace_to_check > 0:
            if pace_to_check < 120:
                recommendations.append("Consider speaking a bit faster to maintain engagement.")
            elif pace_to_check > 160:
                recommendations.append("Try to slow down your speaking pace for better clarity.")

        if default_metrics.get("silence_ratio", 1.0) > 0.3:
            recommendations.append("Reduce long pauses for better flow.")
        
        if confidence_score < 0.7:
            recommendations.append("Work on vocal consistency and energy to improve confidence.")
        
        if not recommendations:
            recommendations = ["Excellent speech clarity and delivery!"]

        # 3. Compile Results (FLAT STRUCTURE)
        analysis_result = {
            "confidence_score": round(confidence_score, 2),
            "speaking_pace": int(round(default_metrics["speaking_pace_wpm"])), # Mapped to Pydantic
            "filler_word_count": filler_word_count,
            "repetition_count": repetition_count,
            "long_pause_count": int(long_pause_count),
            "silence_ratio": round(default_metrics["silence_ratio"], 2),
            "avg_amplitude": round(default_metrics["rms_mean"], 4), # Mapped to Pydantic
            "pitch_mean": round(default_metrics["pitch_mean"], 1),
            "pitch_std": round(default_metrics["pitch_std"], 1),
            "emotion": default_metrics["emotion"],
            "energy_std": round(default_metrics["rms_std"], 4), # Mapped to Pydantic
            "recommendations": recommendations,
            "transcript": transcript,
            "total_words": total_words, 
            "duration_seconds": round(duration_seconds, 2)
        }

        logger.info("✅ Analysis complete.")
        
        # 4. Clean up the temporary file
        os.remove(temp_audio_file)
        logger.info(f"🗑️ Cleaned up temp file: {temp_audio_file}")
        
        # 5. Return the full result structure
        return analysis_result

    except ClientError as e:
        logger.error(f"❌ S3 Error during worker processing: {e}", exc_info=True)
        raise # Re-raise S3 errors
    except Exception as e:
        logger.error(f"❌ CRITICAL Analysis Error: {e}", exc_info=True)
        # If a CRITICAL error occurs (e.g., permission issue, major corruption)
        # The job fails, but we ensure cleanup.
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
        
        worker = Worker(['default'], connection=redis_conn)
        worker.work()

    except Exception as e:
        logger.error(f"❌ Worker failed to start or connect to Redis: {e}", exc_info=True)
        exit(1)
