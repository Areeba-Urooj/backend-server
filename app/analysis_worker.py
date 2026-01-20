# analysis_worker.py

import os
import logging
import subprocess
import numpy as np
import soundfile as sf
import json

# --- IMPORTS FOR OPENAI (MANUAL HTTP) ---
import httpx
# ------------------------------------

import boto3
from botocore.exceptions import ClientError
from redis import Redis
from rq import Worker 

# Import ALL necessary functions and constants from the engine (UPDATED IMPORTS)
from analysis_engine import ( 
    detect_fillers_and_apologies, # New function
    detect_repetitions_for_highlighting, # New function
    TextMarker # Import new NamedTuple
)

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION', 'eu-north-1') 
TARGET_SR = 16000 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [WORKER] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- S3 Client Initialization (Unchanged) ---
def get_s3_client():
    if not S3_BUCKET_NAME:
        logger.error("[WORKER] ❌ S3_BUCKET_NAME environment variable is not set.")
        raise ValueError("S3_BUCKET_NAME is not configured.")
    return boto3.client('s3', region_name=AWS_REGION)

try:
    s3_client = get_s3_client()
    logger.info(f"[WORKER] ✅ S3 Client initialized for bucket: {S3_BUCKET_NAME} in region: {AWS_REGION}")
except Exception as e:
    logger.error(f"[WORKER] ❌ S3 Initialization failed: {e}")
    raise
    
# --- Global ML Model/Scaler Initialization (Unchanged) ---
try:
    EMOTION_MODEL, EMOTION_SCALER, _ = initialize_emotion_model() 
    logger.info("[WORKER] ✅ Emotion model and scaler initialized/loaded.")
except Exception as e:
    logger.error(f"[WORKER] ❌ Failed to initialize emotion model: {e}", exc_info=True)
    EMOTION_MODEL, EMOTION_SCALER = None, None

# --- Intelligent Feedback Generation Function (Unchanged) ---
def generate_intelligent_feedback(transcript: str, metrics: Dict[str, Any]) -> List[str]:
    """Generates tailored feedback and recommendations using a direct HTTP request."""
    # ... (function body remains unchanged) ...
    if not OPENAI_API_KEY:
        logger.warning("[OPENAI] Skipping feedback generation: OPENAI_API_KEY not set.")
        return ["Error: Feedback service is unavailable (API key not configured)."]
        
    acoustic_count = metrics.get('acoustic_disfluency_count', 0)
    
    metrics_summary = json.dumps({
        "Duration (Seconds)": metrics.get('duration_seconds', 0.0),
        "Total Words": metrics.get('total_words', 0),
        "Pace (Words per Minute)": metrics.get('speaking_pace', 0),
        "Filler Word Count": metrics.get('filler_word_count', 0),
        "Repetition Count": metrics.get('repetition_count', 0),
        "Acoustic Disfluency Count (Blocks/Stutters)": acoustic_count, 
        "Silence Ratio": f"{metrics.get('silence_ratio', 0.0) * 100:.2f}%",
        "Emotion Detected": metrics.get('emotion'),
        "Confidence Score": f"{metrics.get('confidence_score', 0.0):.2f}",
    }, indent=2)
    
    system_prompt = (
        "You are an expert speech coach and a genuine mentor. Your primary goal is to provide encouraging, "
        "personalized, and highly actionable guidance, making the user feel seen and understood, not just analyzed. "
        "Do not use generic phrases like 'great job' or 'keep practicing.' "
        "Your feedback MUST directly reference a specific element from the **Transcript** or **Metrics** to justify the recommendation. "
        "For example, instead of 'Improve pacing,' say 'Your pace was highly consistent for the first 30 seconds; let's maintain that control through the entire minute.' "
        "Generate a structured list of exactly three specific recommendations. "
        "The response MUST be a JSON array of strings wrapped in a single key called 'recommendations' "
        "(e.g., {'recommendations': ['Tip 1', 'Tip 2', 'Tip 3']}). Do not include any introductory text, "
        "closing remarks, or numbering outside the array."
    )
    
    user_prompt = (
        f"Transcript (First 1000 characters):\n---\n{transcript[:1000]}...\n---\n"
        f"Analysis Metrics:\n---\n{metrics_summary}\n---\n"
        "Based on these, generate a JSON object with a single key 'recommendations' containing a list of 3 specific recommendations."
    )

    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
        logger.error(f"[OPENAI] HTTP request error: {e}")
        return [f"An error occurred connecting to the feedback service: {e.__class__.__name__}"]
    except httpx.HTTPStatusError as e:
        logger.error(f"[OPENAI] HTTP status error: {e.response.status_code} - {e.response.text}")
        return [f"Feedback service returned an error: {e.response.status_code}"]
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[OPENAI] API call or JSON decoding failed: {e.__class__.__name__}: {e}")
        return [f"An error occurred during intelligent feedback generation: {e.__class__.__name__}"]

# --- Core Analysis Function (UPDATED) ---
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
    
    # 1. Fluency analysis first
    # FIX: Use a robust split method to avoid errors from punctuation/whitespace
    import re
    word_tokens = re.findall(r'\b\w+\b', transcript)
    total_words = len(word_tokens) 
    
    # Collect all text markers from the updated functions
    filler_apology_markers: List[TextMarker] = detect_fillers_and_apologies(transcript)
    repetition_markers: List[TextMarker] = detect_repetitions_for_highlighting(transcript)
    custom_markers: List[TextMarker] = detect_custom_markers(transcript)
    
    all_text_markers = filler_apology_markers + repetition_markers + custom_markers
    
    # Calculate counts for metrics based on the markers
    filler_word_count = len([m for m in all_text_markers if m.type == 'filler'])
    repetition_count = len([m for m in all_text_markers if m.type == 'repetition']) 
    apology_count = len([m for m in all_text_markers if m.type == 'apology']) 
    
    # Initialize values
    duration_seconds = 0.0
    speaking_pace_wpm = 0.0
    emotion = "Neutral"
    confidence_score = 0.0
    
    try:
        # 2. Download (Original raw file)
        logger.info(f"⬇️ Downloading s3://{S3_BUCKET_NAME}/{s3_key}...")
        s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_audio_file)
        
        # 3. Safe Duration Check on the raw file
        audio_features_raw = extract_audio_features(temp_audio_file)
        audio_duration = audio_features_raw.get('duration_s', 0.0)
        
        if audio_duration > MAX_DURATION_SECONDS:
             logger.warning(f"⚠️ Audio duration {audio_duration:.2f}s exceeds limit of {MAX_DURATION_SECONDS}s.")
        
        # 4. Conversion to WAV 
        ffmpeg_command = ["ffmpeg", "-i", temp_audio_file, "-ac", "1", "-ar", str(TARGET_SR), "-y", temp_wav_file]
        
        try:
            subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
            logger.info("✅ FFmpeg conversion successful.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FFmpeg conversion failed: {e.stderr}", exc_info=True)
            raise Exception("FFmpeg conversion failed.")
        except FileNotFoundError:
            logger.error("❌ FFmpeg command not found. Ensure FFmpeg is installed and in PATH.")
            raise Exception("FFmpeg not available in the worker environment.")
            
        # 5. Load the converted WAV file 
        y, sr = sf.read(temp_wav_file, dtype='float32', always_2d=False)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        total_frames = len(rms_frames)
        silence_ratio = silence_frames / total_frames if total_frames > 0 else 0.0
        logger.info(f"📊 Silence ratio: {silence_ratio:.2%} ({silence_frames}/{total_frames} frames)")
        
        # FIX for long_pause_count: Calculate the actual count of DISTINCT long pause events
        long_pause_duration_frames = int(0.5 / (hop_len / sr)) # 0.5 second pause minimum
        
        is_silence = rms_frames < rms_threshold
        long_pause_count = 0
        in_long_pause = False
        pause_start_frame = -1

        for i in range(len(is_silence)):
            if is_silence[i]:
                if not in_long_pause:
                    in_long_pause = True
                    pause_start_frame = i
            elif in_long_pause:
                # End of silence segment
                pause_duration_frames = i - pause_start_frame
                if pause_duration_frames >= long_pause_duration_frames:
                    long_pause_count += 1
                in_long_pause = False

        # Check for pause at the very end
        if in_long_pause and (len(is_silence) - pause_start_frame) >= long_pause_duration_frames:
             long_pause_count += 1
        # END FIX

        # Uses the new NumPy/SciPy function
        pitch_mean, pitch_std = calculate_pitch_stats(y, sr)
        logger.info(f"📊 Pitch stats: mean={pitch_mean:.1f}Hz, std={pitch_std:.1f}Hz")
        
        audio_features_for_score = {
            "rms_mean": float(rms),
            "rms_std": float(energy_std), 
            "speaking_pace_wpm": speaking_pace_wpm,
            "pitch_std": float(pitch_std),
            "pitch_mean": float(pitch_mean),
        }
        
        # Uses the new NumPy/SciPy function
        acoustic_disfluencies = detect_acoustic_disfluencies(y, sr)
        serializable_disfluencies = [d._asdict() for d in acoustic_disfluencies]
        
        # Recalculate fluency metrics for scoring
        fluency_metrics_for_score = {
            "filler_word_count": filler_word_count,
            "repetition_count": repetition_count,
            "acoustic_disfluency_count": len(serializable_disfluencies),
            "total_words": total_words,
        }
        
        confidence_score = score_confidence(audio_features_for_score, fluency_metrics_for_score)
        
        # Emotion Classification
        emotion = classify_emotion_simple(temp_wav_file, EMOTION_MODEL, EMOTION_SCALER)
        
        # 7. Compile Core Metrics for LLM (UPDATED)
        core_analysis_metrics = {
            # Core metrics
            "confidence_score": round(confidence_score, 2),
            "speaking_pace": int(round(speaking_pace_wpm)),  # ✅ MUST be here
            "total_words": total_words,  # ✅ MUST be here
            "duration_seconds": round(duration_seconds, 2),

            # Fluency metrics
            "filler_word_count": filler_word_count,  # ✅ MUST be here
            "repetition_count": repetition_count,  # ✅ MUST be here
            "apology_count": apology_count,

            # Acoustic metrics
            "long_pause_count": int(long_pause_count),  # ✅ MUST be int
            "silence_ratio": round(silence_ratio, 4),

            # Audio features
            "pitch_mean": round(float(pitch_mean), 2),
            "pitch_std": round(float(pitch_std), 2),
            "avg_amplitude": round(float(rms), 6),
            "energy_std": round(float(energy_std), 4),

            # Other
            "emotion": emotion.lower(),
            "acoustic_disfluency_count": len(serializable_disfluencies),
            "transcript": transcript,

            # 🔥 CRITICAL: Include transcript_markers for highlighting
            "transcript_markers": [m._asdict() for m in all_text_markers],
        }

        # 8. Generate Intelligent Feedback
        logger.info("🤖 Generating intelligent feedback using OpenAI...")
        llm_recommendations = generate_intelligent_feedback(
            transcript=transcript,
            metrics=core_analysis_metrics
        )

        # 9. Compile COMPLETE Final Result with ALL metrics
        final_result = {
            # ===== CORE METRICS =====
            "confidence_score": round(confidence_score, 2),
            "speaking_pace": int(round(speaking_pace_wpm)),  # 🔥 MUST be here
            "total_words": total_words,  # ✅ Already working
            "duration_seconds": round(duration_seconds, 2),


            # ===== ANALYSIS DETAILS =====
            "emotion": emotion.lower(),
            "acoustic_disfluency_count": len(serializable_disfluencies),

            # ===== TEXT & HIGHLIGHTING =====
            "transcript": transcript,
            "highlight_markers": [m._asdict() for m in all_text_markers],
            "transcript_markers": [m._asdict() for m in all_text_markers],

        logger.info(f"✅ Analysis complete in {round(time.time() - job_start_time, 2)}s.")
        
        # 10. Cleanup
        if os.path.exists(temp_audio_file): os.remove(temp_audio_file)
        if os.path.exists(temp_wav_file): os.remove(temp_wav_file)
        
        return final_result

    except Exception as e:
        logger.error(f"❌ FATAL Analysis Error: {e}", exc_info=True)
        if os.path.exists(temp_audio_file): os.remove(temp_audio_file)
        if os.path.exists(temp_wav_file): os.remove(temp_wav_file)
        raise

# --- Worker Entrypoint (Unchanged) ---
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
