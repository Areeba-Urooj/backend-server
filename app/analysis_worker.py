import os
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import librosa
import numpy as np
import scipy.signal as signal
import redis
from rq import Worker # Import the necessary RQ components

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

# --- Audio Analysis Service Logic (Unchanged from previous successful version) ---

class AudioAnalysisService:
    def __init__(self):
        self.filler_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in FILLER_WORDS) + r')\b',
            re.IGNORECASE
        )

    def analyze_audio(self, file_path: str, transcript: Optional[str] = None) -> Dict[str, Any]:
        try:
            # ... (librosa loading and feature extraction logic remains the same) ...
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            audio_features = self._extract_audio_features(y, sr)

            transcript_analysis = {}
            if transcript:
                transcript_analysis = self._analyze_transcript(transcript)
                total_words = transcript_analysis.get('total_words', 0)
                speaking_pace = (total_words / duration) * 60 if duration > 0 else 0
                audio_features['speaking_pace'] = speaking_pace
            else:
                audio_features['speaking_pace'] = self._estimate_speaking_pace(y, sr)

            confidence_score = self._calculate_confidence_score(
                audio_features,
                transcript_analysis.get('filler_word_analysis', {}),
                transcript_analysis.get('repetition_count', 0)
            )
            emotion = self._determine_emotion(audio_features)
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
                    "sample_rate": audio_features.get('sample_rate', 0),
                    "channels": audio_features.get('channels', 0),
                    "rms_mean": audio_features.get('rms_mean', 0.0),
                    "rms_std": audio_features.get('rms_std', 0.0),
                    "pitch_mean": audio_features.get('pitch_mean', 0.0),
                    "pitch_std": audio_features.get('pitch_std', 0.0),
                    "pitch_min": audio_features.get('pitch_min', 0.0),
                    "pitch_max": audio_features.get('pitch_max', 0.0),
                    "speaking_pace": audio_features.get('speaking_pace', 0.0),
                    "silence_ratio": audio_features.get('silence_ratio', 0.0),
                    "zcr_mean": audio_features.get('zcr_mean', 0.0),
                },
                "filler_word_analysis": transcript_analysis.get('filler_word_analysis', {}),
                "repetition_count": transcript_analysis.get('repetition_count', 0),
                "long_pause_count": transcript_analysis.get('long_pause_count', 0),
                "total_words": transcript_analysis.get('total_words', 0),
                "confidence_score": confidence_score,
                "emotion": emotion,
                "pitch_variation_score": audio_features.get('pitch_variation_score', 0.0),
                "recommendations": recommendations,
                "analyzed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing audio in worker: {e}", exc_info=True)
            raise

    # --- Private methods are unchanged and are omitted for brevity, assume they are present ---
    # def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, float]: ...
    # def _analyze_transcript(self, transcript: str) -> Dict[str, Any]: ...
    # def _estimate_speaking_pace(self, y: np.ndarray, sr: int) -> float: ...
    # def _calculate_confidence_score(self, audio_features: Dict[str, float], filler_analysis: Dict[str, Any], repetition_count: int) -> float: ...
    # def _determine_emotion(self, audio_features: Dict[str, float]) -> str: ...
    # def _generate_recommendations(self, audio_features: Dict[str, float], filler_analysis: Dict[str, Any], confidence_score: float, emotion: str, repetition_count: int) -> List[str]: ...


def perform_analysis_job(file_id: str, file_path: str, transcript: str) -> dict:
    """
    The main job function executed by the RQ worker.
    """
    audio_analysis_service = AudioAnalysisService()
    
    logger.info(f"[WORKER] Starting analysis for file_id: {file_id}")
    
    try:
        analysis_result = audio_analysis_service.analyze_audio(
            file_path=file_path,
            transcript=transcript
        )
        
        analysis_result["file_id"] = file_id
        analysis_result["file_name"] = file_id 
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"[WORKER] Job {file_id} failed: {e}", exc_info=True)
        raise 

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
    except Exception as e:
        print(f"[WORKER] FATAL: Could not connect to Redis: {e}")
        exit(1)

    worker = Worker(['default'], connection=conn)
    worker.work()
