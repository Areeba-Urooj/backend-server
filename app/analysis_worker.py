import os
import logging
from typing import Dict, Any, Optional, List
import time
import sys
import subprocess
import numpy as np
import soundfile as sf
import json

# --- NEW IMPORTS FOR OPENAI ---
from openai import OpenAI, RateLimitError, APIError 
# ------------------------------

import boto3
from botocore.exceptions import ClientError
from redis import Redis
from rq import Worker 

# Import ALL necessary functions and constants from the engine
# Assuming 'analysis_engine' is in the Python path
from analysis_engine import ( 
    detect_fillers, 
    detect_repetitions, 
    score_confidence, 
    initialize_emotion_model,
    classify_emotion_simple,
    calculate_pitch_stats 
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
except Exception as e:
    logger.error(f"[WORKER] ❌ S3 Initialization failed: {e}")
    raise
    
# --- Global ML Model/Scaler Initialization ---
try:
    # Assuming initialize_emotion_model returns (MODEL, SCALER, ENCODER)
    EMOTION_MODEL, EMOTION_SCALER, _ = initialize_emotion_model() 
    logger.info("[WORKER] ✅ Emotion model and scaler initialized/loaded.")
except Exception as e:
    logger.error(f"[WORKER] ❌ Failed to initialize emotion model: {e}", exc_info=True)
    EMOTION_MODEL, EMOTION_SCALER = None, None

# --- OpenAI Client Initialization (FINAL FIXED ATTEMPT) ---
OPENAI_CLIENT = None
try:
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        
        # 🔥 CRITICAL FIX: Temporarily remove all proxy-related environment variables 
        # to definitively bypass the 'unexpected keyword argument proxies' error 
        # caused by environment injection (e.g., from Render).
        
        # List and pop all proxy variables (case-insensitive check)
        original_proxies = {}
        keys_to_remove = [k for k in os.environ.keys() if 'proxy' in k.lower()]
        for k in keys_to_remove:
            original_proxies[k] = os.environ.pop(k)

        # 1. Initialize the client using the API key. 
        OPENAI_CLIENT = OpenAI(api_key=openai_key) 
        
        # 2. Restore original proxy values (optional, but good practice)
        os.environ.update(original_proxies)

        logger.info("[WORKER] ✅ OpenAI Client initialized.")
    else:
        logger.warning("[WORKER] ⚠️ OPENAI_API_KEY environment variable not found. Skipping LLM initialization.")
        
except Exception as e:
    # If it fails here, the worker continues, but the LLM feature is disabled.
    logger.error(f"[WORKER] ❌ Failed to initialize OpenAI client: {e}. Worker will continue without LLM features.")
    OPENAI_CLIENT = None
# --------------------------------------------

# --- Intelligent Feedback Generation Function ---
def generate_intelligent_feedback(transcript: str, metrics: Dict[str, Any]) -> List[str]:
    """Generates tailored feedback and recommendations using the OpenAI API."""
    if not OPENAI_CLIENT:
        logger.warning("[OPENAI] Skipping feedback generation: OpenAI client not initialized.")
        return ["Error: Feedback service unavailable. Please check the OPENAI_API_KEY environment variable and ensure the worker started correctly."]

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
        "The response MUST be a JSON array of strings wrapped in a single key called 'recommendations' "
        "(e.g., {'recommendations': ['Tip 1', 'Tip 2', 'Tip 3']}). Do not include any introductory text, "
        "closing remarks, or numbering outside the array."
    )
    
    user_prompt = (
        f"Transcript (First 1000 characters):\n---\n{transcript[:1000]}...\n---\n"
        f"Analysis Metrics:\n---\n{metrics_summary}\n---\n"
        "Based on these, generate a JSON object with a single key 'recommendations' containing a list of 3 specific recommendations."
    )

    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.7
        )
        
        feedback_json_str = response.choices[0].message.content
        feedback_data = json.loads(feedback_json_str)
        recommendations = feedback_data.get('recommendations', []) 
        
        if isinstance(recommendations, list) and all(isinstance(r, str) for r in recommendations):
            logger.info(f"[OPENAI] Successfully generated {len(recommendations)} recommendations.")
            return recommendations
        else:
            logger.error(f"[OPENAI] Generated feedback was not a list of strings: {feedback_data}")
            return ["Feedback generation failed: Model returned incorrect format. (Check LLM instructions)"]

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
    transcript: str, 
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Worker function to fetch audio, perform analysis, and return results."""
    job_start_time = time.time()
    logger.info(f"🚀 Starting analysis for file_id: {file_id}, s3_key: {s3_key}")

    temp_audio_file = f"/tmp/{file_id}_{os.path.basename(s3_key)}"
    temp_wav_file = f"/tmp/{file_id}_converted.wav"
    
    # 1. Fluency analysis first (uses the raw transcript immediately)
    total_words = len(transcript.split()) 
    filler_word_count = detect_fillers(transcript)
    repetition_count = detect_repetitions(transcript)
    
    # Initialize values
    duration_seconds = 0
    speaking_pace_wpm = 0
    emotion = "Neutral"
    confidence_score = 0.0
    
    # We will compute the rest of the metrics inside the try block
    try:
        # 2. Download and Convert (Existing FFmpeg Logic)
        logger.info(f"⬇️ Downloading s3://{S3_BUCKET_NAME}/{s3_key}...")
        s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_audio_file)
        
        ffmpeg_command = ["ffmpeg", "-i", temp_audio_file, "-ac", "1", "-ar", str(TARGET_SR), "-y", temp_wav_file]
        
        # Check if ffmpeg is available and run the command
        try:
            subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
            logger.info("✅ FFmpeg conversion successful.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FFmpeg conversion failed: {e.stderr}", exc_info=True)
            raise Exception("FFmpeg conversion failed.")
        except FileNotFoundError:
            logger.error("❌ FFmpeg command not found. Ensure FFmpeg is installed and in PATH.")
            raise Exception("FFmpeg not available in the worker environment.")
            

        # 3. Load the converted WAV file
        y, sr = sf.read(temp_wav_file)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        duration_seconds = len(y) / sr
        speaking_pace_wpm = (total_words / max(1, duration_seconds)) * 60

        # 4. Feature Extraction 
        logger.info("📈 Extracting audio features and metrics...")
        
        rms = np.sqrt(np.mean(y**2)) 
        avg_amplitude = np.mean(np.abs(y))
        energy_std = np.std(np.abs(y)) 
        
        # Simple silence ratio/pause count approximation
        y_abs = np.abs(y)
        amplitude_threshold = np.mean(y_abs) * 0.2
        silence_ratio = np.sum(y_abs < amplitude_threshold) / len(y_abs)
        long_pause_count = float(int(silence_ratio * (duration_seconds / 5))) 
        
        pitch_mean_proxy, pitch_std_proxy = calculate_pitch_stats(y, sr)
        
        pitch_min = 0.0 # Placeholder
        pitch_max = 0.0 # Placeholder
        
        audio_features = {
            "rms_mean": float(rms),
            "rms_std": float(energy_std), 
            "speaking_pace_wpm": speaking_pace_wpm,
            "pitch_std": float(pitch_std_proxy),
            "pitch_mean": float(pitch_mean_proxy),
        }
        fluency_metrics = {
            "filler_word_count": filler_word_count,
            "repetition_count": repetition_count,
            "total_words": total_words,
        }
        
        confidence_score = score_confidence(audio_features, fluency_metrics)
        emotion = classify_emotion_simple(temp_wav_file, EMOTION_MODEL, EMOTION_SCALER)

        # 5. Compile Core Metrics for LLM
        core_analysis_metrics = {
            "confidence_score": round(confidence_score, 2),
            "speaking_pace": int(round(speaking_pace_wpm)),
            "filler_word_count": filler_word_count,
            "repetition_count": repetition_count,
            "long_pause_count": long_pause_count,
            "silence_ratio": round(silence_ratio, 2),
            "avg_amplitude": round(float(avg_amplitude), 4),
            "pitch_mean": round(float(pitch_mean_proxy), 2),
            "pitch_std": round(float(pitch_std_proxy), 2),
            "pitch_min": pitch_min, # Placeholder
            "pitch_max": pitch_max, # Placeholder
            "emotion": emotion.lower(),
            "energy_std": round(float(energy_std), 4),
            "transcript": transcript,
            "total_words": total_words,
            "duration_seconds": round(duration_seconds, 2),
        }
        
        # 6. Generate Intelligent Feedback
        logger.info("🤖 Generating intelligent feedback using OpenAI...")
        llm_recommendations = generate_intelligent_feedback(
            transcript=transcript, 
            metrics=core_analysis_metrics
        )
        
        # 7. Compile Final Result
        final_result = {
            **core_analysis_metrics,
            "recommendations": llm_recommendations,
        }

        logger.info(f"✅ Analysis complete in {round(time.time() - job_start_time, 2)}s.")
        
        # 8. Cleanup
        if os.path.exists(temp_audio_file): os.remove(temp_audio_file)
        if os.path.exists(temp_wav_file): os.remove(temp_wav_file)
        
        return final_result

    except Exception as e:
        logger.error(f"❌ FATAL Analysis Error: {e}", exc_info=True)
        # Final cleanup attempt before re-raising
        if os.path.exists(temp_audio_file): os.remove(temp_audio_file)
        if os.path.exists(temp_wav_file): os.remove(temp_wav_file)
        raise

# --- Worker Entrypoint ---
if __name__ == '__main__':
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    logger.info(f"Starting worker and connecting to Redis at: {redis_url}")
    
    try:
        redis_conn = Redis.from_url(redis_url)
        redis_conn.ping()
        logger.info("✅ Redis connection established")
        
        worker = Worker(['default'], connection=redis_conn)
        worker.work()

    except Exception as e:
        logger.error(f"❌ Worker failed to start: {e}", exc_info=True)
        if 'redis' in str(e).lower():
            logger.critical("⚠️ Check your REDIS_URL and network configuration.")
        exit(1)
