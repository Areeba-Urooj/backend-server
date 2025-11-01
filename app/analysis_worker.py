import os
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
import soundfile as sf  # Replace librosa with soundfile
from scipy import signal
from scipy.fft import fft

# Import ML/Fluency Logic
from app.analysis_engine import (
    detect_fillers, 
    detect_repetitions, 
    score_confidence, 
    initialize_emotion_model,
    classify_emotion
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
    
    # Pad signal
    y_padded = np.pad(y, int(frame_length // 2), mode='reflect')
    
    # Calculate RMS for each frame
    rms = []
    for i in range(0, len(y_padded) - frame_length, hop_length):
        frame = y_padded[i:i + frame_length]
        rms.append(np.sqrt(np.mean(frame**2)))
    
    return np.array(rms)

def extract_pitch(y, sr):
    """Simple pitch estimation using autocorrelation."""
    # Use a smaller segment for faster processing
    segment_length = min(len(y), sr * 2)  # Max 2 seconds
    y_segment = y[:segment_length]
    
    # Compute autocorrelation
    correlation = np.correlate(y_segment, y_segment, mode='full')
    correlation = correlation[len(correlation)//2:]
    
    # Find peaks
    min_period = int(sr / 300)  # Max 300 Hz
    max_period = int(sr / 75)   # Min 75 Hz
    
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
        raise RuntimeError("Emotion model or scaler is not loaded. Cannot perform ML analysis.")

    try:
        # 1. Download from S3
        logger.info(f"⬇️ Downloading s3://{S3_BUCKET_NAME}/{s3_key} to {temp_audio_file}")
        s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_audio_file)
        logger.info("✅ Download complete.")
        
        # 2. Load audio using soundfile (NO LIBROSA)
        logger.info("🎵 Loading audio file...")
        y, sr = sf.read(temp_audio_file)
        
        # Convert stereo to mono if needed
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        duration_seconds = len(y) / sr
        logger.info(f"📊 Audio duration: {duration_seconds:.2f}s, Sample rate: {sr} Hz")
        
        # 3. Extract audio features (without librosa)
        logger.info("📈 Extracting audio features...")
        
        # RMS energy
        rms = extract_rms(y)
        avg_amplitude = np.mean(np.abs(y))
        energy_std = np.std(rms)
        
        # Silence analysis
        rms_threshold = np.mean(rms) * 0.2
        silence_ratio = np.sum(rms < rms_threshold) / len(rms) if len(rms) > 0 else 0
        long_pause_count = int(silence_ratio * (duration_seconds / 5))
        
        # Pitch analysis (simplified)
        logger.info("🎼 Analyzing pitch...")
        pitch_mean = extract_pitch(y, sr)
        # For pitch_std, we'd need multiple pitch estimates which is expensive
        # Use a simple approximation based on amplitude variation
        pitch_std = energy_std * 50  # Rough approximation
        
        # Speaking pace
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
        
        # 5. Emotion Classification
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

def classify_emotion_simple(audio_path: str, model, scaler) -> str:
    """Simplified emotion classification without librosa."""
    try:
        # Load audio
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Limit duration for speed
        max_samples = sr * 30  # 30 seconds max
        if len(y) > max_samples:
            y = y[:max_samples]
        
        # Extract simple features
        # 1. MFCCs approximation using FFT and mel filterbank (simplified)
        n_fft = 2048
        hop_length = 512
        n_mfcc = 13
        
        # Simple spectral features instead of true MFCCs
        frame_count = min(100, (len(y) - n_fft) // hop_length)
        mfcc_approx = []
        
        for i in range(frame_count):
            start = i * hop_length
            frame = y[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            
            # FFT
            spectrum = np.abs(fft(frame))[:n_fft//2]
            
            # Log scale
            log_spectrum = np.log(spectrum + 1e-10)
            
            # Take first 13 coefficients as pseudo-MFCC
            mfcc_approx.append(log_spectrum[:n_mfcc])
        
        mfccs_scaled = np.mean(mfcc_approx, axis=0) if mfcc_approx else np.zeros(n_mfcc)
        
        # 2. Spectral centroid (simplified)
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        spectrum_full = np.abs(fft(y[:n_fft]))[:len(freqs)]
        spectral_centroid = np.sum(freqs * spectrum_full) / (np.sum(spectrum_full) + 1e-10)
        
        # 3. Spectral rolloff
        cumsum = np.cumsum(spectrum_full)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else sr/2
        
        # 4. Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / 2
        zcr = zero_crossings / len(y)
        
        # Combine features
        features = np.concatenate([
            mfccs_scaled,
            [spectral_centroid],
            [spectral_rolloff],
            [zcr]
        ])
        
        # Scale and predict
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error classifying emotion for {audio_path}: {e}", exc_info=True)
        return "Neutral"

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
