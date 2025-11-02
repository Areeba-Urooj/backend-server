import os
# We no longer need Numba/Librosa, so these ENV vars are not necessary,
# but they don't hurt anything if left in.
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['LIBROSA_USE_NATIVE_MPG123'] = '1'

import logging
from typing import Dict, Any, Optional, List
import time
import sys
import subprocess
import numpy as np
import soundfile as sf
import json # <-- NEW: Import for JSON handling

# --- NEW IMPORTS FOR OPENAI ---
from openai import OpenAI, RateLimitError, APIError 
# ------------------------------

from botocore.exceptions import ClientError
from redis import Redis
from rq import Worker

# Import ML/Fluency Logic
from app.analysis_engine import (
    detect_fillers, 
    detect_repetitions, 
    score_confidence, 
    initialize_emotion_model,
    classify_emotion_simple
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

# --- S3 Client Initialization (Existing Code) ---
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
    
# --- Global ML Model/Scaler Initialization (Existing Code) ---
try:
    EMOTION_MODEL, EMOTION_SCALER, _ = initialize_emotion_model()
    logger.info("[WORKER] ✅ Emotion model and scaler initialized/loaded.")
except Exception as e:
    logger.error(f"[WORKER] ❌ Failed to initialize emotion model: {e}", exc_info=True)
    EMOTION_MODEL, EMOTION_SCALER = None, None

# --- OpenAI Client Initialization ---
# This client will automatically pick up the OPENAI_API_KEY environment variable.
try:
    OPENAI_CLIENT = OpenAI()
    logger.info("[WORKER] ✅ OpenAI Client initialized.")
except Exception as e:
    logger.error(f"[WORKER] ❌ Failed to initialize OpenAI client: {e}")
    OPENAI_CLIENT = None
# ------------------------------------

# --- Audio Processing Functions (Existing Code) ---
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
    
# --- NEW: Intelligent Feedback Generation Function ---
def generate_intelligent_feedback(transcript: str, metrics: Dict[str, Any]) -> List[str]:
    """
    Generates tailored feedback and recommendations using the OpenAI API.
    """
    if not OPENAI_CLIENT:
        logger.warning("[OPENAI] Skipping feedback generation: OpenAI client not initialized.")
        return ["Error: Feedback service unavailable. Please check the OPENAI_API_KEY environment variable."]

    # Format the input data for the LLM
    metrics_summary = json.dumps({
        "Duration (Seconds)": metrics.get('duration_seconds', 0.0),
        "Total Words": metrics.get('total_words', 0),
        "Pace (Words per Minute)": metrics.get('speaking_pace', 0),
        "Filler Word Count": metrics.get('filler_word_count', 0),
        "Repetition Count": metrics.get('repetition_count', 0),
        "Silence Ratio": f"{metrics.get('silence_ratio', 0.0) * 100:.2f}%",
        "Emotion Detected": metrics.get('emotion'),
        "Confidence Score": f"{metrics.get('confidence_score', 0.0):.2f}",
    }, indent=2)
    
    system_prompt = (
        "You are an expert speech coach. Your task is to analyze the provided speech transcript and metrics. "
        "Generate a structured list of exactly three highly specific, actionable, and encouraging recommendations "
        "for the user to improve their public speaking. "
        "The response MUST be a JSON array of strings (e.g., ['Tip 1', 'Tip 2', 'Tip 3']). "
        "Do not include any introductory text, closing remarks, or numbering."
    )
    
    user_prompt = (
        f"Transcript:\n---\n{transcript[:1000]}...\n---\n" # Limit transcript length for prompt
        f"Analysis Metrics:\n---\n{metrics_summary}\n---\n"
        "Based on these, generate a JSON array of 3 specific recommendations."
    )

    try:
        logger.info("[OPENAI] Calling Chat API for feedback generation...")
        
        # Use gpt-4o-mini for speed and cost-effectiveness with structured output
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, # Request JSON output
            temperature=0.7
        )
        
        feedback_json_str = response.choices[0].message.content
        feedback_data = json.loads(feedback_json_str)
        
        # The model is instructed to return an array of strings inside a JSON,
        # so we will look for a key like 'recommendations' or just assume the array is the value.
        # Since we asked for a JSON object response_format, we must parse the object:
        recommendations = feedback_data.get('recommendations', []) 
        
        if isinstance(recommendations, list) and all(isinstance(r, str) for r in recommendations):
            logger.info(f"[OPENAI] Successfully generated {len(recommendations)} recommendations.")
            return recommendations
        else:
            # Fallback if the model returns a slightly incorrect JSON structure
            if 'recommendations' not in feedback_data:
                 # Try to extract a list of strings if the model returned { "feedback": [...] }
                 for key, value in feedback_data.items():
                    if isinstance(value, list) and all(isinstance(r, str) for r in value):
                        return value
            
            logger.error(f"[OPENAI] Generated feedback was not a list of strings: {feedback_data}")
            return ["Feedback generation failed: Model returned incorrect format. (Check LLM logs)"]

    except RateLimitError:
        logger.error("[OPENAI] Rate limit exceeded for feedback generation.")
        return ["Feedback service is temporarily busy. Please try again later."]
    except (APIError, json.JSONDecodeError, Exception) as e:
        logger.error(f"[OPENAI] API call or JSON decoding failed: {e.__class__.__name__}: {e}")
        return [f"An error occurred during intelligent feedback generation: {e.__class__.__name__}"]

# --- Core Analysis Function ---
def perform_analysis_job(
    file_id: str, 
    s3_key: str, 
    transcript: str, # 🟢 NOTE 1: This is the raw transcript from the Flutter app
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Worker function to fetch audio, perform analysis, and return results."""
    job_start_time = time.time()
    logger.info(f"🚀 Starting analysis for file_id: {file_id}, s3_key: {s3_key}")
    if user_id:
        logger.info(f"User ID: {user_id}")

    temp_audio_file = f"/tmp/{file_id}_{os.path.basename(s3_key)}"
    temp_wav_file = f"/tmp/{file_id}_converted.wav"
    duration_seconds = 0
    
    # 🟢 NOTE 2: Use the raw transcript to calculate the total words
    total_words = len(transcript.split()) 

    if EMOTION_MODEL is None or EMOTION_SCALER is None:
        logger.warning("[WORKER] ⚠️ Emotion model not loaded, using fallback")

    try:
        # 1-3. Download, Convert, and Load (Existing Code)
        logger.info(f"⬇️ Downloading s3://{S3_BUCKET_NAME}/{s3_key} to {temp_audio_file}")
        s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_audio_file)
        logger.info("✅ Download complete.")
        
        logger.info(f"🎬 Converting M4A ({temp_audio_file}) to WAV ({temp_wav_file}) using FFmpeg...")
        ffmpeg_command = [
            "ffmpeg", 
            "-i", temp_audio_file,
            "-ac", "1",
            "-ar", str(TARGET_SR),
            "-y",
            temp_wav_file
        ]
        
        try:
            result = subprocess.run(ffmpeg_command, 
                                    capture_output=True, 
                                    text=True, 
                                    check=True)
            logger.info("✅ FFmpeg conversion successful.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FFmpeg Conversion Error. Is FFmpeg installed via apt.txt?")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("❌ FFmpeg command not found.")
            raise

        logger.info(f"🎵 Loading converted WAV file: {temp_wav_file}")
        y, sr = sf.read(temp_wav_file)
        
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        duration_seconds = len(y) / sr
        logger.info(f"📊 Audio duration: {duration_seconds:.2f}s, Sample rate: {sr} Hz")
        
        # 4. Extract audio features (Existing Code)
        logger.info("📈 Extracting audio features...")
        
        rms = extract_rms(y)
        avg_amplitude = np.mean(np.abs(y))
        energy_std = np.std(rms)
        
        rms_threshold = np.mean(rms) * 0.2
        silence_ratio = np.sum(rms < rms_threshold) / len(rms) if len(rms) > 0 else 0
        long_pause_count = int(silence_ratio * (duration_seconds / 5))
        
        logger.info("🎼 Analyzing pitch...")
        pitch_mean = extract_pitch(y, sr)
        pitch_std = energy_std * 50
        
        speaking_pace_wpm = (total_words / max(1, duration_seconds)) * 60

        # 5. Fluency Analysis (Existing Code)
        logger.info("💬 Analyzing transcript fluency...")
        # 🟢 NOTE 3: Fluency functions use the raw transcript (as desired)
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
        
        # 6. Emotion Classification (Existing Code)
        logger.info("😊 Classifying emotion...")
        emotion = classify_emotion_simple(temp_wav_file, EMOTION_MODEL, EMOTION_SCALER)

        # 7. Compile Core Metrics
        # This dictionary is used both for the final result AND the LLM prompt
        core_analysis_metrics = {
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
            "transcript": transcript, # Retain the raw transcript
            "total_words": total_words,
            "duration_seconds": round(duration_seconds, 2), # CRITICAL for Flutter app
        }
        
        # 8. 🧠 NEW STEP: Generate Intelligent Feedback
        logger.info("🤖 Generating intelligent feedback using OpenAI...")
        llm_recommendations = generate_intelligent_feedback(
            transcript=transcript, 
            metrics=core_analysis_metrics
        )
        
        # 9. Compile Final Result (Merge core metrics with LLM recommendations)
        final_result = {
            **core_analysis_metrics,
            "recommendations": llm_recommendations,
        }

        logger.info(f"✅ Analysis complete in {round(time.time() - job_start_time, 2)}s. Emotion: {emotion}, Confidence: {confidence_score}")
        
        # 10. Cleanup (Existing Code)
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
            logger.info(f"🗑️ Cleaned up temp M4A file: {temp_audio_file}")
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
            logger.info(f"🗑️ Cleaned up converted WAV file: {temp_wav_file}")
        
        return final_result

    except ClientError as e:
        logger.error(f"❌ S3 Error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Analysis Error: {e}", exc_info=True)
        raise
    finally:
        # Final cleanup attempt
        if os.path.exists(temp_audio_file):
             os.remove(temp_audio_file)
        if os.path.exists(temp_wav_file):
             os.remove(temp_wav_file)

# --- Worker Entrypoint (Existing Code) ---
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
