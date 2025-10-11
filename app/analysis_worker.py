import os
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import librosa
import numpy as np
import scipy.signal as signal
import redis
from rq import Worker
from rq.job import Job

# --- Configuration (CRITICAL: Must be consistent with file_upload_service.py and main.py) ---
UPLOAD_DIR = "uploads"
FILLER_WORDS = [
    "ah", "actually", "almighty", "almost", "and", "anyways", "basically",
    "believe me", "er", "erm", "essentially", "etc", "exactly",
    "for what it's worth", "fwiw", "gosh", "i mean", "i guess", "i suppose",
    "i think", "innit", "isn't it", "just", "like", "literally", "look", "man",
    "my gosh", "oh", "ok", "okay", "see", "so", "tbh", "to be honest",
    "totally", "truly", "uh", "uh-huh", "uhm", "um", "well", "whatever",
    "you know", "you see"
]

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Audio Analysis Service Logic ---

class AudioAnalysisService:
    def __init__(self):
        self.filler_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in FILLER_WORDS) + r')\b',
            re.IGNORECASE
        )

    def analyze_audio(self, file_path: str, transcript: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Load audio and calculate basic duration
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            # 1. Call feature extraction method (now defined)
            audio_features = self._extract_audio_features(y, sr)

            transcript_analysis = {}
            if transcript:
                # 2. Call transcript analysis method (now defined)
                transcript_analysis = self._analyze_transcript(transcript)
                total_words = transcript_analysis.get('total_words', 0)
                speaking_pace = (total_words / duration) * 60 if duration > 0 else 0
                audio_features['speaking_pace'] = speaking_pace
            else:
                # 3. Call pace estimation method (now defined)
                audio_features['speaking_pace'] = self._estimate_speaking_pace(y, sr)

            # 4. Call confidence score method (now defined)
            confidence_score = self._calculate_confidence_score(
                audio_features,
                transcript_analysis.get('filler_word_analysis', {}),
                transcript_analysis.get('repetition_count', 0)
            )
            # 5. Call emotion determination method (now defined)
            emotion = self._determine_emotion(audio_features)
            # 6. Call recommendations generation method (now defined)
            recommendations = self._generate_recommendations(
                audio_features,
                transcript_analysis.get('filler_word_analysis', {}),
                confidence_score,
                emotion,
                transcript_analysis.get('repetition_count', 0)
            )

            # Manually create the nested dictionary structure for the response
            return {
                "duration_seconds": duration,
                "audio_features": {
                    "duration_seconds": duration,
                    # NOTE: Placing placeholder/default values for all features
                    "sample_rate": sr, 
                    "channels": y.ndim, # 1 for mono, 2 for stereo
                    "rms_mean": audio_features.get('rms_mean', 0.1),
                    "rms_std": audio_features.get('rms_std', 0.05),
                    "pitch_mean": audio_features.get('pitch_mean', 120.0),
                    "pitch_std": audio_features.get('pitch_std', 10.0),
                    "pitch_min": audio_features.get('pitch_min', 80.0),
                    "pitch_max": audio_features.get('pitch_max', 180.0),
                    "speaking_pace": audio_features.get('speaking_pace', 150.0),
                    "silence_ratio": audio_features.get('silence_ratio', 0.05),
                    "zcr_mean": audio_features.get('zcr_mean', 0.01),
                },
                "filler_word_analysis": transcript_analysis.get('filler_word_analysis', {'like': 0, 'um': 0, 'total': 0}),
                "repetition_count": transcript_analysis.get('repetition_count', 0),
                "long_pause_count": transcript_analysis.get('long_pause_count', 0),
                "total_words": transcript_analysis.get('total_words', 0),
                "confidence_score": confidence_score,
                "emotion": emotion,
                "pitch_variation_score": audio_features.get('pitch_variation_score', 0.75),
                "recommendations": recommendations,
                "analyzed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing audio in worker: {e}", exc_info=True)
            # CRITICAL: Re-raise the exception so RQ marks the job as FAILED
            raise


    # --------------------------------------------------------------------------
    # CRITICAL FIX: Placeholder methods to ensure the code executes successfully
    # --------------------------------------------------------------------------

    def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Placeholder for actual feature extraction logic."""
        logger.info("Running placeholder _extract_audio_features.")
        
        # In a real implementation, you would calculate: rms_mean, pitch_mean, silence_ratio, etc.
        # For now, we return default/dummy values to avoid an AttributeError.
        return {
            'rms_mean': 0.1,
            'rms_std': 0.05,
            'pitch_mean': 120.0,
            'pitch_std': 10.0,
            'pitch_min': 80.0,
            'pitch_max': 180.0,
            'silence_ratio': 0.05,
            'zcr_mean': 0.01,
            'pitch_variation_score': 0.75,
        }

    def _analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """Placeholder for transcript analysis (filler words, word count)."""
        logger.info("Running placeholder _analyze_transcript.")
        
        words = transcript.split()
        total_words = len(words)
        
        # Simple filler word count
        filler_matches = self.filler_word_pattern.findall(transcript)
        filler_counts = {word.lower(): filler_matches.count(word.lower()) for word in set(filler_matches)}
        filler_counts['total'] = len(filler_matches)

        return {
            'filler_word_analysis': filler_counts,
            'repetition_count': 0, # Placeholder
            'long_pause_count': 0, # Placeholder (This should be from audio features)
            'total_words': total_words
        }

    def _estimate_speaking_pace(self, y: np.ndarray, sr: int) -> float:
        """Placeholder for pace estimation without a transcript."""
        logger.info("Running placeholder _estimate_speaking_pace.")
        return 150.0 # Default WPM

    def _calculate_confidence_score(self, audio_features: Dict[str, float], filler_analysis: Dict[str, Any], repetition_count: int) -> float:
        """Placeholder for the final confidence calculation."""
        logger.info("Running placeholder _calculate_confidence_score.")
        # Logic would involve weighting metrics (e.g., filler words, pace, pitch)
        return 85.0 # High default score

    def _determine_emotion(self, audio_features: Dict[str, float]) -> str:
        """Placeholder for emotion detection."""
        logger.info("Running placeholder _determine_emotion.")
        return "Neutral"

    def _generate_recommendations(self, audio_features: Dict[str, float], filler_analysis: Dict[str, Any], confidence_score: float, emotion: str, repetition_count: int) -> List[str]:
        """Placeholder for generating feedback."""
        logger.info("Running placeholder _generate_recommendations.")
        return ["Try to reduce filler words.", "Maintain your current pace."]


def perform_analysis_job(file_id: str, file_path: str, transcript: str) -> dict:
    # ... (rest of the perform_analysis_job function is correct) ...
    """
    The main job function executed by the RQ worker.
    """
    audio_analysis_service = AudioAnalysisService()
    
    # Check for file existence *just before* running the analysis
    if not os.path.exists(file_path):
        logger.error(f"[WORKER] CRITICAL: File not found at {file_path}. Job failed.")
        # Raise a specific error to help diagnosis
        raise FileNotFoundError(f"Audio file not found for analysis: {file_path}")

    logger.info(f"[WORKER] Starting analysis for file_id: {file_id}. File size: {os.path.getsize(file_path)} bytes.")
    
    try:
        analysis_result = audio_analysis_service.analyze_audio(
            file_path=file_path,
            transcript=transcript
        )
        
        # Note: Added file_name here to match the expected AnalysisStatusResponse.result structure if needed
        analysis_result["file_id"] = file_id
        analysis_result["file_name"] = os.path.basename(file_path)
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"[WORKER] Job {file_id} failed with an unhandled exception: {e}", exc_info=True)
        raise # Re-raise the exception so RQ marks the job as FAILED
    
    finally:
        # Cleanup the file *after* analysis (success or failure)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"[WORKER] Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"[WORKER] Failed to delete file {file_path}: {e}")
        

# ----------------------------------------------------
# --- RQ Worker Startup Script ---
# ----------------------------------------------------
if __name__ == '__main__':
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    try:
        conn = redis.from_url(redis_url)
        conn.ping()
        print(f"[WORKER] Connected to Redis at: {redis_url}. Starting worker...")
        
        # ✅ CORRECT: Initialize worker AFTER successful connection
        worker = Worker(['default'], connection=conn)
        worker.work()
        
    except Exception as e:
        print(f"[WORKER] FATAL: Could not connect to Redis: {e}")
        exit(1)
