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
    # Use the explicitly set AWS_REGION
    return boto3.client('s3', region_name=AWS_REGION)

try:
    s3_client = get_s3_client()
    logger.info(f"[WORKER] ✅ S3 Client initialized for bucket: {S3_BUCKET_NAME} in region: {AWS_REGION}")
except ValueError as e:
    logger.error(f"[WORKER] ❌ Configuration Error: {e}")
    raise

# --- Core Analysis Function ---
def perform_analysis_job(
    file_id: str, 
    s3_key: str, 
    transcript: str, 
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Worker function to fetch audio, perform analysis, and return the results.
    Includes placeholder analysis logic using librosa and numpy.
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
        rms_threshold = np.mean(rms) * 0.2  # 20% of mean RMS
        silence_ratio = np.sum(rms < rms_threshold) / len(rms)
        long_pause_count = int(silence_ratio * 10) # Placeholder metric
        
        # Simple Placeholder for Pitch (F0)
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=300)
        pitch_valid = pitches[magnitudes > np.quantile(magnitudes, 0.9)]
        pitch_mean = np.mean(pitch_valid) if len(pitch_valid) > 0 else 0
        pitch_std = np.std(pitch_valid) if len(pitch_valid) > 0 else 0
        
        # Speaking Pace (words per minute)
        speaking_pace_wpm = (total_words / max(1, duration_seconds)) * 60

        # Placeholder for filler words and repetitions (requires NLP/transcript processing)
        filler_word_count = transcript.lower().count('um') + transcript.lower().count('uh')
        repetition_count = 2 # Hardcoded placeholder

        # 3. Compile Results
        analysis_result = {
            "duration_seconds": round(duration_seconds, 2),
            "total_words": total_words,
            "repetition_count": repetition_count,
            "long_pause_count": long_pause_count,
            "confidence_score": round(np.random.uniform(0.9, 0.99), 2), # Random confidence
            "emotion": "calm", # Hardcoded placeholder
            "recommendations": ["Ensure clear articulation.", "Try to reduce filler words."] if filler_word_count > 0 else ["Excellent pace and clarity."],
            "audio_features": {
                "rms_mean": round(np.mean(rms), 4),
                "rms_std": round(np.std(rms), 4),
                "silence_ratio": round(silence_ratio, 2),
                "speaking_pace_wpm": round(speaking_pace_wpm, 1),
                "pitch_mean": round(pitch_mean, 1),
                "pitch_std": round(pitch_std, 1),
            },
            "filler_word_analysis": {
                "filler_word_count": filler_word_count,
                "filler_word_rate": round(filler_word_count / total_words, 3) if total_words > 0 else 0,
            },
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

# --- Worker Entrypoint ---
if __name__ == '__main__':
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    logger.info(f"Starting worker and connecting to Redis at: {redis_url}")
    
    try:
        redis_conn = Redis.from_url(redis_url)
        redis_conn.ping()
        logger.info("Redis connection established.")
        
        # ✅ FINAL FIX: Pass the connection object directly to the Worker constructor
        worker = Worker(['default'], connection=redis_conn)
        worker.work()

    except Exception as e:
        logger.error(f"❌ Worker failed to start or connect to Redis: {e}", exc_info=True)
        exit(1)
