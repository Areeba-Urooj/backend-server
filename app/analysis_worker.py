import os
# These ENV vars are for librosa, we can leave them in, but they won't interfere
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['LIBROSA_USE_NATIVE_MPG123'] = '1'

import logging
from typing import Dict, Any, Optional
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import boto3
from botocore.exceptions import ClientError
from redis import Redis
from rq import Worker 
import numpy as np
import soundfile as sf
from scipy.fft import fft
from pydub import AudioSegment # <-- NEW: Import for M4A handling
from pydub.exceptions import CouldntDecodeError # <-- NEW: Import for error handling

# Import ML/Fluency Logic - FIXED IMPORT
from app.analysis_engine import (
    detect_fillers, 
    detect_repetitions, 
    score_confidence, 
    initialize_emotion_model,
    classify_emotion_simple  # CHANGED: Was classify_emotion
)

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION', 'eu-north-1') 
TARGET_SR = 16000 # Standard sample rate for speech analysis

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [WORKER] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- S3 Client Initialization ---
def get_s3_client():
    """Initializes and returns the S3 client."""
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

# --- Global ML Model/Scaler Initialization ---
try:
    EMOTION_MODEL, EMOTION_SCALER, _ = initialize_emotion_model()
    logger.info("[WORKER] ✅ Emotion model and scaler initialized/loaded.")
except Exception as e:
    logger.error(f"[WORKER] ❌ Failed to initialize emotion model: {e}", exc_info=True)
    EMOTION_MODEL, EMOTION_SCALER = None, None

# --- Audio Processing Functions (Replacing Librosa) ---
def extract_rms(y):
    """Calculate RMS (Root Mean Square) energy."""
    frame_length = 2048
    hop_length = 512
    
    y_padded = np.pad(y, int(frame_length // 2), mode='reflect')
    
    rms = []
    for i in range(0, len(y_padded) - frame_length, hop_length):
        frame = y_padded[i:i + frame_length]
        rms.append(np.sqrt(np.mean(frame**2)))
    
    return np.array(rms)

def extract_pitch(y, sr):
    """Simple pitch estimation using autocorrelation."""
    segment_length = min(len(y), sr * 2)
    y_segment = y[:segment_length]
    
    correlation = np.correlate(y_segment, y_segment, mode='full')
    correlation = correlation[len(correlation)//2:]
    
    min_period = int(sr / 300)
    max_period = int(sr / 75)
    
    if max_period < len(correlation):
        peak_region = correlation[min_period:max_period]
        if len(peak_region) > 0:
            peak_idx = np.argmax(peak_region) + min_period
            pitch = sr / peak_idx
            return pitch
    
    return 0

# --- Core Analysis Function ---
def perform_analysis_job(
    file_id: str, 
    s3_key: str, 
    transcript: str, 
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Worker function to fetch audio, perform analysis, and return results."""
    job_start_time = time.time()
    logger.info(f"🚀 Starting analysis for file_id: {file_id}, s3_key: {s3_key}")
    if user_id:
        logger.info(f"User ID: {user_id}")

    temp_audio_file = f"/tmp/{file_id}_{os.path.basename(s3_key)}"
    duration_seconds = 0
    total_words = len(transcript.split())

    if EMOTION_MODEL is None or EMOTION_SCALER is None:
        logger.warning("[WORKER] ⚠️ Emotion model not loaded, using fallback")

    try:
        # 1. Download from S3
        logger.info(f"⬇️ Downloading s3://{S3_BUCKET_NAME}/{s3_key} to {temp_audio_file}")
        s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_audio_file)
        logger.info("✅ Download complete.")
        
        # 2. Load audio using pydub for M4A support <-- FIX APPLIED HERE
        logger.info(f"🎵 Loading audio file using pydub (M4A format expected)...")
        
        audio = None
        try:
            # Load the audio using pydub. FFmpeg is used to decode the M4A file.
            audio = AudioSegment.from_file(temp_audio_file, format="m4a")
            
            # Resample and convert to mono for consistent analysis
            if audio.frame_rate != TARGET_SR:
                audio = audio.set_frame_rate(TARGET_SR)
            if audio.channels > 1:
                audio = audio.set_channels(1)
                
            sr = TARGET_SR
            
            # Convert to normalized NumPy array (float32, range [-1.0, 1.0])
            y = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
            
            duration_seconds = audio.duration_seconds
            logger.info(f"📊 Audio duration: {duration_seconds:.2f}s, Sample rate: {sr} Hz (Processed by pydub)")
            
        except CouldntDecodeError as e:
            logger.error(f"❌ Pydub Decode Error: Ensure FFmpeg is installed and accessible. {e}", exc_info=True)
            raise Exception("Failed to decode audio file using pydub/FFmpeg. Format not supported.") from e
        except Exception as e:
            # Fallback to soundfile for other formats/errors (though highly unlikely for M4A)
            logger.warning(f"Pydub failed, trying soundfile as fallback. Error: {e}")
            y, sr = sf.read(temp_audio_file)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            duration_seconds = len(y) / sr


        # 3. Extract audio features
        logger.info("📈 Extracting audio features...")
        
        rms = extract_rms(y)
        avg_amplitude = np.mean(np.abs(y))
        energy_std = np.std(rms)
        
        rms_threshold = np.mean(rms) * 0.2
        silence_ratio = np.sum(rms < rms_threshold) / len(rms) if len(rms) > 0 else 0
        long_pause_count = int(silence_ratio * (duration_seconds / 5))
        
        logger.info("🎼 Analyzing pitch...")
        pitch_mean = extract_pitch(y, sr)
        # Pitch standard deviation is approximated based on energy stability
        pitch_std = energy_std * 50
        
        speaking_pace_wpm = (total_words / max(1, duration_seconds)) * 60

        # 4. Fluency Analysis
        logger.info("💬 Analyzing transcript fluency...")
        filler_word_count = detect_fillers(transcript)
        repetition_count = detect_repetitions(transcript)
        
        audio_features = {
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "speaking_pace_wpm": speaking_pace_wpm,
            "pitch_std": float(pitch_std),
            "pitch_mean": float(pitch_mean),
        }
        fluency_metrics = {
            "filler_word_count": filler_word_count,
            "repetition_count": repetition_count,
            "total_words": total_words,
        }
        
        confidence_score = score_confidence(audio_features, fluency_metrics)
        
        # 5. Emotion Classification - FIXED FUNCTION CALL
        logger.info("😊 Classifying emotion...")
        emotion = classify_emotion_simple(temp_audio_file, EMOTION_MODEL, EMOTION_SCALER)

        # 6. Compile Results
        analysis_result = {
            "confidence_score": round(confidence_score, 2),
            "speaking_pace": int(round(speaking_pace_wpm)),
            "filler_word_count": filler_word_count,
            "repetition_count": repetition_count,
            "long_pause_count": float(long_pause_count),
            "silence_ratio": round(silence_ratio, 2),
            "avg_amplitude": round(float(avg_amplitude), 4),
            "pitch_mean": round(float(pitch_mean), 2),
            "pitch_std": round(float(pitch_std), 2),
            "emotion": emotion.lower(),
            "energy_std": round(float(energy_std), 4),
            "recommendations": [
                f"Your speaking pace is {int(round(speaking_pace_wpm))} WPM. Consider adjusting it toward 140 WPM.",
                f"You used {filler_word_count} filler words. Focus on concise pauses.",
                f"Your emotional tone was classified as **{emotion}**."
            ],
            "transcript": transcript,
        }

        logger.info(f"✅ Analysis complete in {round(time.time() - job_start_time, 2)}s. Emotion: {emotion}, Confidence: {confidence_score}")
        
        # 7. Cleanup
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
            logger.info(f"🗑️ Cleaned up temp file: {temp_audio_file}")
        
        return analysis_result

    except ClientError as e:
        logger.error(f"❌ S3 Error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Analysis Error: {e}", exc_info=True)
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
        
        worker = Worker(['default'], connection=redis_conn)
        worker.work()

    except Exception as e:
        logger.error(f"❌ Worker failed to start: {e}", exc_info=True)
        exit(1)
