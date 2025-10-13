# app/analysis_worker.py

import os
import logging
from typing import Dict, Any, Optional
import time

import boto3
from botocore.exceptions import ClientError
from redis import Redis
# ✅ FIX 1: Revert import to the lowercase 'connections' submodule name
from rq import connections, Worker 

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
    """
    job_start_time = time.time()
    logger.info(f"🚀 Starting analysis for file_id: {file_id}, s3_key: {s3_key}")
    if user_id:
        logger.info(f"User ID: {user_id}")

    temp_audio_file = f"/tmp/{file_id}.m4a"

    try:
        # 1. Download the file from S3
        logger.info(f"⬇️ Downloading s3://{S3_BUCKET_NAME}/{s3_key} to {temp_audio_file}")
        s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_audio_file)
        logger.info("✅ Download complete.")

        # --- 2. Perform Placeholder Analysis ---
        total_words = len(transcript.split())
        speaking_pace = total_words / max(1, (time.time() - job_start_time))
        
        analysis_result = {
            "duration_seconds": 15.5,
            "total_words": total_words,
            "repetition_count": 2,
            "long_pause_count": 1,
            "confidence_score": 0.95,
            "emotion": "calm",
            "recommendations": ["Speak slightly faster.", "Vary your pitch."],
            "audio_features": {
                "rms_mean": 0.05,
                "silence_ratio": 0.10,
                "speaking_pace": speaking_pace,
                "pitch_mean": 120.0,
                "pitch_std": 10.0,
                "pitch_min": 80.0,
                "pitch_max": 180.0,
                "rms_std": 0.015,
            },
            "filler_word_analysis": {
                "filler_word_count": 5,
                "filler_word_rate": 0.05,
            },
            "transcript": transcript,
        }

        logger.info("✅ Analysis complete.")
        
        # 3. Clean up the temporary file
        os.remove(temp_audio_file)
        logger.info(f"🗑️ Cleaned up temp file: {temp_audio_file}")
        
        # 4. Return the full result structure
        return analysis_result

    except ClientError as e:
        logger.error(f"❌ S3 Error during worker processing: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ General Analysis Error: {e}", exc_info=True)
        raise
    finally:
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
        
        # ✅ FIX 2: Set the default connection explicitly before starting the worker.
        connections.set_default_connection(redis_conn)
        
        worker = Worker(['default'])
        worker.work()

    except Exception as e:
        logger.error(f"❌ Worker failed to start or connect to Redis: {e}", exc_info=True)
        exit(1)
