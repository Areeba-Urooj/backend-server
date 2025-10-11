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

# --- Configuration ---
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

            # Extract audio features
            audio_features = self._extract_audio_features(y, sr)

            transcript_analysis = {}
            if transcript:
                transcript_analysis = self._analyze_transcript(transcript)
                total_words = transcript_analysis.get('total_words', 0)
                speaking_pace = (total_words / duration) * 60 if duration > 0 else 0
                audio_features['speaking_pace'] = speaking_pace
            else:
                audio_features['speaking_pace'] = self._estimate_speaking_pace(y, sr)

            # Calculate metrics
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

            # Build response structure
            return {
                "duration_seconds": duration,
                "audio_features": {
                    "duration_seconds": duration,
                    "sample_rate": sr, 
                    "channels": y.ndim,
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
            raise

    def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Placeholder for actual feature extraction logic."""
        logger.info("Running placeholder _extract_audio_features.")
        
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
        """Placeholder for transcript analysis."""
        logger.info("Running placeholder _analyze_transcript.")
        
        words = transcript.split()
        total_words = len(words)
        
        filler_matches = self.filler_word_pattern.findall(transcript)
        filler_counts = {word.lower(): filler_matches.count(word.lower()) for word in set(filler_matches)}
        filler_counts['total'] = len(filler_matches)

        return {
            'filler_word_analysis': filler_counts,
            'repetition_count': 0,
            'long_pause_count': 0,
            'total_words': total_words
        }

    def _estimate_speaking_pace(self, y: np.ndarray, sr: int) -> float:
        """Placeholder for pace estimation without a transcript."""
        logger.info("Running placeholder _estimate_speaking_pace.")
        return 150.0

    def _calculate_confidence_score(self, audio_features: Dict[str, float], filler_analysis: Dict[str, Any], repetition_count: int) -> float:
        """Placeholder for confidence calculation."""
        logger.info("Running placeholder _calculate_confidence_score.")
        return 85.0

    def _determine_emotion(self, audio_features: Dict[str, float]) -> str:
        """Placeholder for emotion detection."""
        logger.info("Running placeholder _determine_emotion.")
        return "Neutral"

    def _generate_recommendations(self, audio_features: Dict[str, float], filler_analysis: Dict[str, Any], confidence_score: float, emotion: str, repetition_count: int) -> List[str]:
        """Placeholder for generating feedback."""
        logger.info("Running placeholder _generate_recommendations.")
        return ["Try to reduce filler words.", "Maintain your current pace."]


def perform_analysis_job(file_id: str, file_path: str, transcript: str) -> dict:
    """
    The main job function executed by the RQ worker.
    """
    audio_analysis_service = AudioAnalysisService()
    
    # Check for file existence before analysis
    if not os.path.exists(file_path):
        logger.error(f"[WORKER] CRITICAL: File not found at {file_path}. Job failed.")
        raise FileNotFoundError(f"Audio file not found for analysis: {file_path}")

    logger.info(f"[WORKER] Starting analysis for file_id: {file_id}. File size: {os.path.getsize(file_path)} bytes.")
    
    try:
        analysis_result = audio_analysis_service.analyze_audio(
            file_path=file_path,
            transcript=transcript
        )
        
        analysis_result["file_id"] = file_id
        analysis_result["file_name"] = os.path.basename(file_path)
        
        logger.info(f"[WORKER] Successfully completed analysis for {file_id}")
        return analysis_result
    
    except Exception as e:
        logger.error(f"[WORKER] Job {file_id} failed with exception: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup the file after analysis
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
    import sys
    
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    print(f"[WORKER] Attempting to connect to Redis at: {redis_url}")
    
    try:
        conn = redis.from_url(redis_url)
        conn.ping()
        print(f"[WORKER] ✓ Connected to Redis successfully")
        print(f"[WORKER] ✓ Initializing worker for 'default' queue...")
        
        # ✅ CRITICAL FIX: Initialize and start worker AFTER successful Redis connection
        worker = Worker(['default'], connection=conn)
        print(f"[WORKER] ✓ Worker initialized, starting to listen for jobs...")
        
        # This is the blocking call that keeps the worker running
        worker.work(logging_level='INFO')
        
    except KeyboardInterrupt:
        print("\n[WORKER] Worker stopped by user (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        print(f"[WORKER] ✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
