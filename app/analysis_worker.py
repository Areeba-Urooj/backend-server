import os
import sys
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add proper path handling for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Audio Analysis Service Logic ---

class AudioAnalysisService:
    def __init__(self):
        self.filler_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in FILLER_WORDS) + r')\b',
            re.IGNORECASE
        )

    def analyze_audio(self, file_path: str, transcript: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"[ANALYSIS] Starting audio analysis for: {file_path}")
        
        try:
            # Load audio and calculate basic duration
            logger.info(f"[ANALYSIS] Loading audio file...")
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            logger.info(f"[ANALYSIS] Audio loaded: duration={duration:.2f}s, sample_rate={sr}Hz")

            # Extract audio features
            logger.info(f"[ANALYSIS] Extracting audio features...")
            audio_features = self._extract_audio_features(y, sr)

            transcript_analysis = {}
            if transcript:
                logger.info(f"[ANALYSIS] Analyzing transcript ({len(transcript)} chars)...")
                transcript_analysis = self._analyze_transcript(transcript)
                total_words = transcript_analysis.get('total_words', 0)
                speaking_pace = (total_words / duration) * 60 if duration > 0 else 0
                audio_features['speaking_pace'] = speaking_pace
                logger.info(f"[ANALYSIS] Speaking pace: {speaking_pace:.1f} WPM")
            else:
                logger.info(f"[ANALYSIS] No transcript provided, estimating pace...")
                audio_features['speaking_pace'] = self._estimate_speaking_pace(y, sr)

            # Calculate metrics
            logger.info(f"[ANALYSIS] Calculating confidence score...")
            confidence_score = self._calculate_confidence_score(
                audio_features,
                transcript_analysis.get('filler_word_analysis', {}),
                transcript_analysis.get('repetition_count', 0)
            )
            
            logger.info(f"[ANALYSIS] Determining emotion...")
            emotion = self._determine_emotion(audio_features)
            
            logger.info(f"[ANALYSIS] Generating recommendations...")
            recommendations = self._generate_recommendations(
                audio_features,
                transcript_analysis.get('filler_word_analysis', {}),
                confidence_score,
                emotion,
                transcript_analysis.get('repetition_count', 0)
            )

            # Build response structure
            result = {
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
                "filler_word_analysis": transcript_analysis.get('filler_word_analysis', {'total': 0}),
                "repetition_count": transcript_analysis.get('repetition_count', 0),
                "long_pause_count": transcript_analysis.get('long_pause_count', 0),
                "total_words": transcript_analysis.get('total_words', 0),
                "confidence_score": confidence_score,
                "emotion": emotion,
                "pitch_variation_score": audio_features.get('pitch_variation_score', 0.75),
                "recommendations": recommendations,
                "analyzed_at": datetime.now().isoformat()
            }
            
            logger.info(f"[ANALYSIS] ✅ Analysis completed successfully")
            return result

        except Exception as e:
            logger.error(f"[ANALYSIS] ❌ Error analyzing audio: {e}", exc_info=True)
            raise

    def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract audio features from the waveform."""
        logger.debug("[ANALYSIS] Extracting audio features (placeholder implementation)")
        
        try:
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            
            # Calculate zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(np.mean(zcr))
            
            return {
                'rms_mean': rms_mean,
                'rms_std': rms_std,
                'pitch_mean': 120.0,  # Placeholder
                'pitch_std': 10.0,
                'pitch_min': 80.0,
                'pitch_max': 180.0,
                'silence_ratio': 0.05,
                'zcr_mean': zcr_mean,
                'pitch_variation_score': 0.75,
            }
        except Exception as e:
            logger.warning(f"[ANALYSIS] Feature extraction error (using defaults): {e}")
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
        """Analyze transcript for filler words and other metrics."""
        logger.debug("[ANALYSIS] Analyzing transcript")
        
        words = transcript.split()
        total_words = len(words)
        
        # Count filler words
        filler_matches = self.filler_word_pattern.findall(transcript.lower())
        filler_counts = {}
        for word in set(filler_matches):
            filler_counts[word] = filler_matches.count(word)
        filler_counts['total'] = len(filler_matches)
        
        logger.debug(f"[ANALYSIS] Found {len(filler_matches)} filler words in {total_words} total words")

        return {
            'filler_word_analysis': filler_counts,
            'repetition_count': 0,  # Placeholder
            'long_pause_count': 0,  # Placeholder
            'total_words': total_words
        }

    def _estimate_speaking_pace(self, y: np.ndarray, sr: int) -> float:
        """Estimate speaking pace without transcript."""
        logger.debug("[ANALYSIS] Estimating speaking pace")
        return 150.0  # Default WPM

    def _calculate_confidence_score(self, audio_features: Dict[str, float], 
                                    filler_analysis: Dict[str, Any], 
                                    repetition_count: int) -> float:
        """Calculate confidence score based on features."""
        logger.debug("[ANALYSIS] Calculating confidence score")
        
        # Simple scoring logic
        base_score = 100.0
        
        # Penalize for filler words
        filler_count = filler_analysis.get('total', 0)
        base_score -= min(filler_count * 2, 20)
        
        return max(min(base_score, 100.0), 0.0)

    def _determine_emotion(self, audio_features: Dict[str, float]) -> str:
        """Determine emotion from audio features."""
        logger.debug("[ANALYSIS] Determining emotion")
        return "Neutral"

    def _generate_recommendations(self, audio_features: Dict[str, float], 
                                  filler_analysis: Dict[str, Any], 
                                  confidence_score: float, 
                                  emotion: str, 
                                  repetition_count: int) -> List[str]:
        """Generate recommendations based on analysis."""
        logger.debug("[ANALYSIS] Generating recommendations")
        
        recommendations = []
        
        filler_count = filler_analysis.get('total', 0)
        if filler_count > 5:
            recommendations.append(f"Try to reduce filler words. You used {filler_count} filler words.")
        else:
            recommendations.append("Great job minimizing filler words!")
        
        pace = audio_features.get('speaking_pace', 150)
        if pace < 120:
            recommendations.append("Consider speaking a bit faster to maintain engagement.")
        elif pace > 180:
            recommendations.append("Try slowing down slightly to ensure clarity.")
        else:
            recommendations.append("Your speaking pace is excellent!")
        
        return recommendations


def perform_analysis_job(file_id: str, file_path: str, transcript: str) -> dict:
    """
    The main job function executed by the RQ worker.
    This function is called by RQ when a job is dequeued.
    """
    logger.info(f"[WORKER] ========================================")
    logger.info(f"[WORKER] 🎯 JOB STARTED: {file_id}")
    logger.info(f"[WORKER] File path: {file_path}")
    logger.info(f"[WORKER] Transcript length: {len(transcript)} chars")
    logger.info(f"[WORKER] ========================================")
    
    # Check for file existence before analysis
    if not os.path.exists(file_path):
        logger.error(f"[WORKER] ❌ CRITICAL: File not found at {file_path}")
        raise FileNotFoundError(f"Audio file not found for analysis: {file_path}")

    file_size = os.path.getsize(file_path)
    logger.info(f"[WORKER] ✅ File exists: {file_size} bytes")
    
    audio_analysis_service = AudioAnalysisService()
    
    try:
        logger.info(f"[WORKER] Starting analysis service...")
        analysis_result = audio_analysis_service.analyze_audio(
            file_path=file_path,
            transcript=transcript
        )
        
        analysis_result["file_id"] = file_id
        analysis_result["file_name"] = os.path.basename(file_path)
        
        logger.info(f"[WORKER] ✅ Analysis completed successfully for {file_id}")
        logger.info(f"[WORKER] Result keys: {list(analysis_result.keys())}")
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"[WORKER] ❌ Job {file_id} failed with exception: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup the file after analysis
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"[WORKER] 🗑️  Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"[WORKER] ⚠️  Failed to delete file {file_path}: {e}")


# ----------------------------------------------------
# --- RQ Worker Startup Script ---
# ----------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 WORKER STARTING UP")
    print("="*60 + "\n")
    
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    print(f"[WORKER] Redis URL: {redis_url}")
    
    try:
        # Connect to Redis
        print(f"[WORKER] Attempting to connect to Redis...")
        conn = redis.from_url(redis_url)
        conn.ping()
        print(f"[WORKER] ✅ Connected to Redis successfully")
        
        # Test if we can access the queue
        from rq import Queue
        test_queue = Queue('default', connection=conn)
        queue_length = len(test_queue)
        print(f"[WORKER] 📊 Current queue length: {queue_length}")
        
        print(f"[WORKER] 🎧 Starting worker listening on 'default' queue...")
        print("="*60 + "\n")
        
        # Initialize and start worker
        worker = Worker(['default'], connection=conn)
        
        # Log worker details
        print(f"[WORKER] Worker ID: {worker.name}")
        print(f"[WORKER] Python path: {sys.path}")
        print(f"[WORKER] Current directory: {os.getcwd()}")
        print("\n[WORKER] 🎬 Worker is now running...\n")
        
        worker.work(logging_level='INFO')
        
    except KeyboardInterrupt:
        print("\n[WORKER] ⏹️  Worker stopped by user.")
    except Exception as e:
        print(f"\n[WORKER] ❌ FATAL: Worker failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
