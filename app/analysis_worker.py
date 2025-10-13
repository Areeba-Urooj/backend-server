import os
import sys
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import io # 💡 NEW: Used to handle file data in memory

import boto3 # 💡 NEW: For S3 access
from botocore.exceptions import ClientError # 💡 NEW: To catch S3 errors

# Add proper path handling for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
import scipy.signal as signal
import redis
from rq import Worker
from rq.job import Job

# --- Configuration ---
# ❌ REMOVED: UPLOAD_DIR is no longer relevant
# UPLOAD_DIR = "uploads"

# 💡 NEW: AWS S3 Configuration
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

FILLER_WORDS = [
# ... (FILLER_WORDS list remains the same) ...
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

# --- S3 Client Initialization ---
def get_s3_client():
    """Initializes and returns the S3 client."""
    if not S3_BUCKET_NAME:
        logger.error("[WORKER] ❌ S3_BUCKET_NAME environment variable is not set.")
        raise ValueError("S3_BUCKET_NAME is not configured.")
    # Boto3 automatically uses AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from env vars
    return boto3.client('s3', region_name=AWS_REGION)

# --- Audio Analysis Service Logic ---

class AudioAnalysisService:
# ... (AudioAnalysisService class methods remain the same) ...

    def __init__(self):
        self.filler_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in FILLER_WORDS) + r')\b',
            re.IGNORECASE
        )

    # 💡 CHANGE: This now accepts a file-like object (BytesIO) instead of a file_path
    def analyze_audio(self, audio_file_object: io.BytesIO, transcript: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"[ANALYSIS] Starting audio analysis from in-memory object.")
        
        try:
            # Load audio and calculate basic duration
            logger.info(f"[ANALYSIS] Loading audio file from buffer...")
            # 💡 CHANGE: Pass the BytesIO object to librosa.load
            y, sr = librosa.load(audio_file_object, sr=None) 
            duration = librosa.get_duration(y=y, sr=sr)
            logger.info(f"[ANALYSIS] Audio loaded: duration={duration:.2f}s, sample_rate={sr}Hz")

            # Extract audio features
# ... (rest of the analyze_audio method is unchanged) ...
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

# ... (Helper methods like _extract_audio_features, _analyze_transcript, etc., remain unchanged) ...
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
                'pitch_mean': 120.0,  # Placeholder
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
            'repetition_count': 0,  # Placeholder
            'long_pause_count': 0,  # Placeholder
            'total_words': total_words
        }

    def _estimate_speaking_pace(self, y: np.ndarray, sr: int) -> float:
        """Estimate speaking pace without transcript."""
        logger.debug("[ANALYSIS] Estimating speaking pace")
        return 150.0  # Default WPM

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


# 💡 MAJOR CHANGE: Rewrite of the main job function
def perform_analysis_job(file_id: str, s3_key: str, transcript: str) -> dict:
    """
    The main job function executed by the RQ worker.
    Downloads the file from S3 to an in-memory buffer, performs analysis, and cleans up S3.
    """
    logger.info(f"[WORKER] ========================================")
    logger.info(f"[WORKER] 🎯 JOB STARTED: {file_id}")
    logger.info(f"[WORKER] S3 Key: {s3_key}")
    logger.info(f"[WORKER] Transcript length: {len(transcript)} chars")
    logger.info(f"[WORKER] ========================================")

    s3_client = get_s3_client()
    audio_buffer = io.BytesIO()
    
    try:
        # 1. DOWNLOAD from S3
        logger.info(f"[WORKER] 📥 Downloading file from S3 bucket '{S3_BUCKET_NAME}'...")
        s3_client.download_fileobj(
            Bucket=S3_BUCKET_NAME, 
            Key=s3_key, 
            Fileobj=audio_buffer
        )
        audio_buffer.seek(0) # Rewind the buffer to the start for reading by librosa
        logger.info(f"[WORKER] ✅ Download complete. Buffer size: {len(audio_buffer.getvalue())} bytes")

        # 2. PERFORM ANALYSIS
        audio_analysis_service = AudioAnalysisService()
        logger.info(f"[WORKER] Starting analysis service...")
        # 💡 CHANGE: Pass the in-memory buffer to the service
        analysis_result = audio_analysis_service.analyze_audio(
            audio_file_object=audio_buffer,
            transcript=transcript
        )
        
        analysis_result["file_id"] = file_id
        analysis_result["s3_key"] = s3_key
        
        logger.info(f"[WORKER] ✅ Analysis completed successfully for {file_id}")
        logger.info(f"[WORKER] Result keys: {list(analysis_result.keys())}")
        
        return analysis_result
    
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.error(f"[WORKER] ❌ CRITICAL: S3 file not found: {s3_key}")
            raise FileNotFoundError(f"S3 object not found for analysis: {s3_key}")
        logger.error(f"[WORKER] ❌ S3 Error during download: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"[WORKER] ❌ Job {file_id} failed with exception: {e}", exc_info=True)
        raise
    
    finally:
        # 3. CLEANUP S3 (CRUCIAL!)
        try:
            logger.info(f"[WORKER] 🗑️  Deleting file from S3: {s3_key}")
            s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
            logger.info(f"[WORKER] ✅ S3 cleanup complete.")
        except Exception as e:
            logger.warning(f"[WORKER] ⚠️  Failed to delete S3 object {s3_key}: {e}")

# ... (RQ Worker Startup Script remains the same) ...
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
        print("\n[WORKER] ⏹️  Worker stopped by user.")
    except Exception as e:
        print(f"\n[WORKER] ❌ FATAL: Worker failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
