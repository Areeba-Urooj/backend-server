# analysis_worker.py

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import time
import sys
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
    detect_custom_markers, # New function
    score_confidence, 
    initialize_emotion_model,
    classify_emotion_simple,
    calculate_pitch_stats, 
    detect_acoustic_disfluencies, 
    extract_audio_features,      
    MAX_DURATION_SECONDS,
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

    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"}, 
        "temperature": 0.7
    }

    try:
        logger.info("[OPENAI] Generating feedback with 30s timeout...")

        # Use timeout on HTTP request to prevent hanging
        with httpx.Client(trust_env=False, timeout=30.0) as client:
            client.proxies = {}

            response = client.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30.0  # 30 second timeout
            )

        response.raise_for_status()

        feedback_data = response.json()
        feedback_json_str = feedback_data['choices'][0]['message']['content']

        recommendation_data = json.loads(feedback_json_str)
        recommendations = recommendation_data.get('recommendations', [])

        if isinstance(recommendations, list) and all(isinstance(r, str) for r in recommendations):
            logger.info(f"[OPENAI] Successfully generated {len(recommendations)} recommendations.")
            return recommendations
        else:
            logger.error(f"[OPENAI] Generated feedback was not a list of strings: {recommendation_data}")
            return ["Feedback generation failed: Model returned incorrect format."]

    except httpx.TimeoutException:
        logger.warning("[OPENAI] Feedback generation timed out (30s). Using fallback.")
        return [
            "Great effort! Keep practicing to reduce filler words.",
            "Work on maintaining a consistent speaking pace.",
            "Your presentation is improving with practice!"
        ]
    except httpx.RequestError as e:
        logger.error(f"[OPENAI] HTTP request error: {e}")
        return [f"An error occurred connecting to the feedback service: {e.__class__.__name__}"]
    except httpx.HTTPStatusError as e:
        logger.error(f"[OPENAI] HTTP status error: {e.response.status_code} - {e.response.text}")
        return [f"Feedback service returned an error: {e.response.status_code}"]
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[OPENAI] API call or JSON decoding failed: {e.__class__.__name__}: {e}")
        return [f"An error occurred during intelligent feedback generation: {e.__class__.__name__}"]

# --- Core Analysis Function (WITH COMPREHENSIVE DEBUG LOGGING) ---
def perform_analysis_job(
    file_id: str,
    s3_key: str,
    transcript: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Worker function to fetch audio, perform analysis, and return results."""

    # Set 10-minute timeout for entire job
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Analysis job exceeded maximum processing time (10 minutes)")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)  # 600 seconds = 10 minutes

    job_start_time = time.time()
    logger.info(f"🚀 [JOB START] file_id={file_id}, s3_key={s3_key}, user_id={user_id}")

    temp_audio_file = f"/tmp/{file_id}_{os.path.basename(s3_key)}"
    temp_wav_file = f"/tmp/{file_id}_converted.wav"

    try:
        # STEP 1: Transcript Analysis
        logger.info("[STEP 1] Starting transcript analysis...")
        try:
            import re
            word_tokens = re.findall(r'\b\w+\b', transcript)
            total_words = len(word_tokens)
            logger.info(f"✅ [STEP 1] Transcript tokenized: {total_words} words found")

            # Fluency analysis
            filler_apology_markers: List[TextMarker] = detect_fillers_and_apologies(transcript)
            logger.info(f"✅ [STEP 1] Fillers/apologies detected: {len(filler_apology_markers)}")

            repetition_markers: List[TextMarker] = detect_repetitions_for_highlighting(transcript)
            logger.info(f"✅ [STEP 1] Repetitions detected: {len(repetition_markers)}")

            custom_markers: List[TextMarker] = detect_custom_markers(transcript)
            logger.info(f"✅ [STEP 1] Custom markers detected: {len(custom_markers)}")

            all_text_markers = filler_apology_markers + repetition_markers + custom_markers
            filler_word_count = len([m for m in all_text_markers if m.type == 'filler'])
            repetition_count = len([m for m in all_text_markers if m.type == 'repetition'])
            apology_count = len([m for m in all_text_markers if m.type == 'apology'])

            logger.info(f"✅ [STEP 1] Complete: fillers={filler_word_count}, reps={repetition_count}, apologies={apology_count}")
        except Exception as e:
            logger.error(f"❌ [STEP 1] FAILED: {e}", exc_info=True)
            raise

        # STEP 2: Download from S3
        logger.info(f"[STEP 2] Downloading S3 file: s3://{S3_BUCKET_NAME}/{s3_key}")
        try:
            s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_audio_file)
            file_size = os.path.getsize(temp_audio_file)
            logger.info(f"✅ [STEP 2] Downloaded successfully: {file_size} bytes")
        except ClientError as e:
            logger.error(f"❌ [STEP 2] S3 download failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"❌ [STEP 2] FAILED: {e}", exc_info=True)
            raise

        # STEP 3: Check duration with ffprobe
        logger.info("[STEP 3] Extracting audio features (ffprobe)...")
        try:
            audio_features_raw = extract_audio_features(temp_audio_file)
            audio_duration = audio_features_raw.get('duration_s', 0.0)
            logger.info(f"✅ [STEP 3] Audio duration: {audio_duration:.2f}s")

            if audio_duration > MAX_DURATION_SECONDS:
                logger.warning(f"⚠️ [STEP 3] Audio exceeds limit: {audio_duration:.2f}s > {MAX_DURATION_SECONDS}s")

            if audio_duration < 1.0:
                logger.error(f"❌ [STEP 3] Audio too short: {audio_duration:.2f}s < 1.0s")
                raise ValueError(f"Audio duration too short: {audio_duration:.2f}s")
        except Exception as e:
            logger.error(f"❌ [STEP 3] FAILED: {e}", exc_info=True)
            raise

        # STEP 4: FFmpeg Conversion (M4A → WAV)
        logger.info("[STEP 4] Converting audio with FFmpeg (M4A → WAV)...")
        try:
            # Use ffmpeg to convert M4A/MP3/etc to WAV with proper settings
            ffmpeg_command = [
                "ffmpeg",
                "-i", temp_audio_file,      # Input: M4A file
                "-ac", "1",                 # Convert to mono
                "-ar", str(TARGET_SR),      # Resample to 16kHz
                "-acodec", "pcm_s16le",     # PCM 16-bit encoding for best compatibility
                "-y",                       # Overwrite output without asking
                temp_wav_file               # Output: WAV file
            ]
            logger.info(f"[STEP 4] Running FFmpeg conversion...")

            result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
            logger.info(f"[STEP 4] FFmpeg conversion completed successfully")

            if not os.path.exists(temp_wav_file):
                raise FileNotFoundError(f"WAV file was not created at {temp_wav_file}")

            wav_size = os.path.getsize(temp_wav_file)
            logger.info(f"✅ [STEP 4] WAV file created successfully: {wav_size} bytes")

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ [STEP 4] FFmpeg conversion failed: {e.stderr}", exc_info=True)
            raise Exception(f"FFmpeg conversion failed: {e.stderr}")
        except FileNotFoundError as e:
            logger.error(f"❌ [STEP 4] {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"❌ [STEP 4] FAILED: {e}", exc_info=True)
            raise

        # STEP 5: Load WAV with soundfile
        logger.info("[STEP 5] Loading WAV file with soundfile...")
        try:
            y, sr = sf.read(temp_wav_file, dtype='float32', always_2d=False)
            logger.info(f"✅ [STEP 5] WAV loaded: shape={y.shape}, sr={sr}Hz")

            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
                logger.info(f"[STEP 5] Converted stereo to mono: new shape={y.shape}")

            # Check audio validity
            if len(y) == 0:
                raise ValueError("Audio array is empty")

            max_samples = sr * MAX_DURATION_SECONDS
            if len(y) > max_samples:
                y = y[:max_samples]
                logger.info(f"[STEP 5] Truncated to {MAX_DURATION_SECONDS}s")

            duration_seconds = len(y) / sr
            logger.info(f"✅ [STEP 5] Audio ready: {duration_seconds:.2f}s, shape={y.shape}")

        except Exception as e:
            logger.error(f"❌ [STEP 5] WAV loading failed: {e}", exc_info=True)
            raise

        # STEP 6: Calculate Speaking Pace
        logger.info("[STEP 6] Calculating speaking pace...")
        try:
            if duration_seconds > 0:
                speaking_pace_wpm = (total_words / duration_seconds) * 60
            else:
                speaking_pace_wpm = 0.0
            logger.info(f"✅ [STEP 6] Speaking pace: {speaking_pace_wpm:.1f} WPM")
        except Exception as e:
            logger.error(f"❌ [STEP 6] FAILED: {e}", exc_info=True)
            raise

        # STEP 7: Extract Audio Features from loaded WAV data
        logger.info("[STEP 7] Extracting audio features from loaded WAV data...")
        try:
            # Calculate RMS energy and other features from the loaded audio data
            # Frame-based feature extraction
            frame_length_feat = int(0.025 * sr)  # 25ms frames
            hop_length_feat = int(0.010 * sr)    # 10ms hop

            # Extract frames
            frames = np.array([
                y[i:i+frame_length_feat]
                for i in range(0, len(y)-frame_length_feat, hop_length_feat)
            ])

            if len(frames) == 0:
                raise ValueError("No frames extracted for feature analysis")

            # Calculate RMS energy per frame
            rms_frames = np.sqrt(np.mean(frames**2, axis=1))
            rms_mean = float(np.mean(rms_frames))
            rms_std = float(np.std(rms_frames))

            # Calculate zero-crossing rate per frame
            zcr_frames = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2
            zcr_mean = float(np.mean(zcr_frames))

            # Store features for later use
            audio_features = {
                'rms_mean': rms_mean,
                'rms_std': rms_std,
                'zcr_mean': zcr_mean,
                'duration_s': duration_seconds,
                'sample_rate': sr
            }

            logger.info(f"✅ [STEP 7] Audio features extracted: RMS={rms_mean:.4f}, std={rms_std:.4f}, ZCR={zcr_mean:.4f}")

        except Exception as e:
            logger.error(f"❌ [STEP 7] FAILED: {e}", exc_info=True)
            raise

        # STEP 8: Calculate Silence and Pauses
        logger.info("[STEP 8] Calculating silence and pauses...")
        try:
            # Calculate silence ratio using proper RMS thresholding
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length_calc = int(0.010 * sr)    # 10ms hop

            # Extract frames for silence analysis
            frames = np.array([
                y[i:i+frame_length]
                for i in range(0, len(y)-frame_length, hop_length_calc)
            ])

            if len(frames) > 0:
                rms_frames = np.sqrt(np.mean(frames**2, axis=1))
                silence_threshold = np.mean(rms_frames) * 0.15  # Robust threshold
                silence_frames = np.sum(rms_frames < silence_threshold)
                total_frames = len(rms_frames)
                silence_ratio = silence_frames / total_frames if total_frames > 0 else 0.0
                logger.info(f"✅ [STEP 8] Silence ratio: {silence_ratio:.2%} ({silence_frames}/{total_frames} frames)")

                # Long pause detection
                min_pause_samples = int(0.5 * sr)  # 0.5 second minimum

                # Find silent regions
                is_silent = rms_frames < silence_threshold
                long_pause_count = 0

                # Count contiguous silent regions longer than minimum duration
                silent_start = None
                for i, silent in enumerate(is_silent):
                    if silent and silent_start is None:
                        silent_start = i
                    elif not silent and silent_start is not None:
                        # Check duration of silent region
                        silent_samples = (i - silent_start) * hop_length_calc
                        if silent_samples >= min_pause_samples:
                            long_pause_count += 1
                        silent_start = None

                # Check for silence at the end
                if silent_start is not None:
                    silent_samples = (len(is_silent) - silent_start) * hop_length_calc
                    if silent_samples >= min_pause_samples:
                        long_pause_count += 1

                logger.info(f"✅ [STEP 8] Long pauses detected: {long_pause_count}")
            else:
                silence_ratio = 0.0
                long_pause_count = 0

        except Exception as e:
            logger.error(f"❌ [STEP 8] FAILED: {e}", exc_info=True)
            raise

        # STEP 9: Calculate Pitch Stats
        logger.info("[STEP 9] Calculating pitch statistics...")
        try:
            pitch_mean, pitch_std = calculate_pitch_stats(y, sr)
            logger.info(f"✅ [STEP 9] Pitch: mean={pitch_mean:.1f}Hz, std={pitch_std:.1f}Hz")
        except Exception as e:
            logger.error(f"❌ [STEP 9] FAILED: {e}", exc_info=True)
            pitch_mean, pitch_std = 185.0, 15.0  # Safe defaults
            logger.warning("[STEP 9] Using fallback pitch values")

        # STEP 10: Detect Acoustic Disfluencies
        logger.info("[STEP 10] Detecting acoustic disfluencies...")
        try:
            acoustic_disfluencies = detect_acoustic_disfluencies(y, sr)
            logger.info(f"✅ [STEP 10] Acoustic disfluencies: {len(acoustic_disfluencies)} detected")
        except Exception as e:
            logger.error(f"❌ [STEP 10] FAILED: {e}", exc_info=True)
            acoustic_disfluencies = []

        # STEP 11: Score Confidence
        logger.info("[STEP 11] Calculating confidence score...")
        try:
            audio_features_for_score = {
                "speaking_pace_wpm": speaking_pace_wpm,
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "energy_std": audio_features['rms_std'],
                "rms_mean": audio_features['rms_mean'],
            }

            fluency_metrics_for_score = {
                "filler_word_count": filler_word_count,
                "repetition_count": repetition_count,
                "acoustic_disfluency_count": len(acoustic_disfluencies),
                "total_words": total_words,
            }

            confidence_score = score_confidence(audio_features_for_score, fluency_metrics_for_score)
            logger.info(f"✅ [STEP 11] Confidence score: {confidence_score:.1f}/100")
        except Exception as e:
            logger.error(f"❌ [STEP 11] FAILED: {e}", exc_info=True)
            raise

        # STEP 12: Emotion Classification
        logger.info("[STEP 12] Classifying emotion...")
        try:
            emotion = classify_emotion_simple(temp_wav_file, EMOTION_MODEL, EMOTION_SCALER)
            logger.info(f"✅ [STEP 12] Emotion: {emotion}")
        except Exception as e:
            logger.error(f"❌ [STEP 12] FAILED: {e}", exc_info=True)
            emotion = "neutral"

        # STEP 13: Compile Final Result with ALL metrics
        logger.info("[STEP 13] Compiling complete analysis result...")
        try:
            final_result = {
                # Core metrics
                "confidence_score": round(confidence_score, 2),
                "speaking_pace": int(round(speaking_pace_wpm)),
                "total_words": total_words,
                "duration_seconds": round(duration_seconds, 2),

                # Fluency metrics
                "filler_word_count": filler_word_count,
                "repetition_count": repetition_count,
                "apology_count": apology_count,

                # Acoustic metrics
                "long_pause_count": int(long_pause_count),
                "silence_ratio": round(silence_ratio, 4),
                "acoustic_disfluency_count": len(acoustic_disfluencies),

                # Audio features
                "pitch_mean": round(float(pitch_mean), 2),
                "pitch_std": round(float(pitch_std), 2),
                "avg_amplitude": round(float(audio_features['rms_mean']), 6),
                "energy_std": round(float(audio_features['rms_std']), 6),

                # Analysis details
                "emotion": emotion.lower(),

                # Text and recommendations
                "transcript": transcript,
                "transcript_markers": [m._asdict() for m in all_text_markers],
            }

            # Log all metrics for debugging
            logger.info(
                f"✅ [STEP 13] Complete result compiled:\n"
                f"  - Confidence: {confidence_score:.1f}/100\n"
                f"  - Speaking Pace: {speaking_pace_wpm:.1f} WPM\n"
                f"  - Total Words: {total_words}\n"
                f"  - Filler Words: {filler_word_count}\n"
                f"  - Repetitions: {repetition_count}\n"
                f"  - Long Pauses: {long_pause_count}\n"
                f"  - Silence Ratio: {silence_ratio:.2%}\n"
                f"  - Pitch: {pitch_mean:.1f}Hz ±{pitch_std:.1f}Hz\n"
                f"  - Energy Std: {audio_features['rms_std']:.6f}"
            )

        except Exception as e:
            logger.error(f"❌ [STEP 13] FAILED: {e}", exc_info=True)
            raise

        # STEP 14: Generate Intelligent Feedback
        logger.info("[STEP 14] Generating AI feedback...")
        try:
            llm_recommendations = generate_intelligent_feedback(
                transcript=transcript,
                metrics=final_result
            )
            final_result["recommendations"] = llm_recommendations
            logger.info(f"✅ [STEP 14] Generated {len(llm_recommendations)} recommendations")
        except Exception as e:
            logger.error(f"❌ [STEP 14] FAILED: {e}", exc_info=True)
            final_result["recommendations"] = ["Unable to generate feedback at this time."]

        # STEP 16: Cleanup
        logger.info("[STEP 16] Cleaning up temporary files...")
        try:
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
                logger.info(f"✅ Removed {temp_audio_file}")
            if os.path.exists(temp_wav_file):
                os.remove(temp_wav_file)
                logger.info(f"✅ Removed {temp_wav_file}")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")

        elapsed = round(time.time() - job_start_time, 2)
        logger.info(f"🏆 [JOB COMPLETE] Analysis finished in {elapsed}s. Confidence={confidence_score:.1f}/100")

        return final_result

    except TimeoutError as e:
        logger.error(f"❌ [JOB TIMEOUT] Analysis took too long: {e}", exc_info=True)

        # Cleanup on timeout
        if os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
            except:
                pass
        if os.path.exists(temp_wav_file):
            try:
                os.remove(temp_wav_file)
            except:
                pass

        raise
    except Exception as e:
        logger.error(f"❌ [JOB FAILED] Fatal error: {e}", exc_info=True)

        # Cleanup on error
        if os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
            except:
                pass
        if os.path.exists(temp_wav_file):
            try:
                os.remove(temp_wav_file)
            except:
                pass

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
