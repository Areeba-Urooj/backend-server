# analysis_worker.py
import os
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import librosa
import numpy as np
import scipy.signal as signal
import redis
from rq import Worker, connections

# --- Configuration (Must be consistent with main.py) ---
UPLOAD_DIR = "uploads_simple"
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

# --- Audio Analysis Service Logic (Copied from main.py) ---

class AudioAnalysisService:
    def __init__(self):
        self.filler_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in FILLER_WORDS) + r')\b',
            re.IGNORECASE
        )

    # NOTE: This is a synchronous function in the worker, which is fine!
    def analyze_audio(self, file_path: str, transcript: Optional[str] = None) -> Dict[str, Any]:
        # NOTE: Made this method synchronous, as it runs in a synchronous worker.
        try:
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
            # This must match the Pydantic schema in main.py for successful return
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

    # Copy all private methods: _extract_audio_features, _analyze_transcript, 
    # _estimate_speaking_pace, _calculate_confidence_score, _determine_emotion, 
    # _generate_recommendations from main.py and paste them here.
    # --- START OF COPIED PRIVATE METHODS ---
    def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        features = {'sample_rate': sr, 'channels': 1 if y.ndim == 1 else y.shape[0]}
        frame_length, hop_length = 2048, 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['silence_ratio'] = float(np.sum(rms < 0.02) / len(rms) if len(rms) > 0 else 0)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=400)
        pitch_values = [pitches[magnitudes[:, i].argmax(), i] for i in range(pitches.shape[1]) if pitches[magnitudes[:, i].argmax(), i] > 0]
        if pitch_values:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_min'] = float(np.min(pitch_values))
            features['pitch_max'] = float(np.max(pitch_values))
            pitch_range = features['pitch_max'] - features['pitch_min']
            features['pitch_variation_score'] = float(min(100, (pitch_range / 50) * 100))
        else:
            features.update({'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_min': 0.0, 'pitch_max': 0.0, 'pitch_variation_score': 0.0})
        return features

    def _analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        words = re.findall(r'\b\w+\b', transcript.lower())
        total_words = len(words)
        filler_matches = self.filler_word_pattern.findall(transcript.lower())
        filler_word_count = len(filler_matches)
        filler_words_dict = {}
        for word in filler_matches:
            filler_words_dict[word] = filler_words_dict.get(word, 0) + 1
        repetition_count = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
        long_pause_count = len(re.findall(r'\.\.\.|\.\.', transcript))
        return {
            'total_words': total_words,
            'filler_word_analysis': {
                'filler_word_count': filler_word_count,
                'filler_words': filler_words_dict,
                'filler_word_ratio': filler_word_count / total_words if total_words > 0 else 0
            },
            'repetition_count': repetition_count,
            'long_pause_count': long_pause_count
        }

    def _estimate_speaking_pace(self, y: np.ndarray, sr: int) -> float:
        energy = librosa.feature.rms(y=y)[0]
        peaks, _ = signal.find_peaks(energy, height=0.1*np.max(energy), distance=sr//4)
        estimated_syllables = len(peaks)
        duration = librosa.get_duration(y=y, sr=sr)
        estimated_words = estimated_syllables / 1.5
        return (estimated_words / duration) * 60 if duration > 0 else 0

    def _calculate_confidence_score(self, audio_features: Dict[str, float], filler_analysis: Dict[str, Any], repetition_count: int) -> float:
        score = 100.0
        
        # Penalty for filler words
        score -= min(30, filler_analysis.get('filler_word_ratio', 0) * 150)
        
        # Penalty for long silences
        if audio_features.get('silence_ratio', 0) > 0.3:
            score -= min(20, (audio_features.get('silence_ratio', 0) - 0.3) * 100)
            
        # Bonus/Penalty for pitch variation
        pitch_variation_score = audio_features.get('pitch_variation_score', 0)
        score += min(15, (pitch_variation_score - 50) / 5) if pitch_variation_score > 50 else -min(20, (50 - pitch_variation_score) / 4)
        
        # Penalty for speaking too fast or too slow
        pace_deviation = abs(audio_features.get('speaking_pace', 150) - 150) / 150
        score -= min(20, pace_deviation * 60)
        
        # Penalty for unstable energy (shaky voice)
        rms_std = audio_features.get('rms_std', 0)
        if rms_std > 0.05:
            score -= min(15, (rms_std - 0.05) * 200)
            
        # Penalty for repetitions
        score -= min(15, repetition_count * 2)

        return max(0, min(100, score))

    def _determine_emotion(self, audio_features: Dict[str, float]) -> str:
        pitch_std = audio_features.get('pitch_std', 0)
        rms_mean = audio_features.get('rms_mean', 0)
        speaking_pace = audio_features.get('speaking_pace', 150)

        if rms_mean > 0.1:
            if pitch_std > 40:
                if speaking_pace > 170:
                    return "excited"
                else:
                    return "energetic"
            elif pitch_std < 15:
                return "monotone"
            else:
                return "neutral"
        elif rms_mean < 0.05:
            if audio_features.get('silence_ratio', 0) > 0.4:
                return "tense"
            else:
                return "calm"
        else:
            if pitch_std > 30:
                return "animated"
            else:
                return "neutral"

    def _generate_recommendations(self, audio_features: Dict[str, float], filler_analysis: Dict[str, Any], confidence_score: float, emotion: str, repetition_count: int) -> List[str]:
        recs = []
        
        # Filler words
        filler_word_count = filler_analysis.get('filler_word_count', 0)
        if filler_word_count > 5:
            common_fillers = sorted(filler_analysis.get('filler_words', {}).items(), key=lambda item: item[1], reverse=True)
            top_fillers = [f'"{item[0]}"' for item in common_fillers[:3]]
            recs.append(f"You used {filler_word_count} filler words. Try to reduce using {', '.join(top_fillers)} to sound more confident.")

        # Repetitions
        if repetition_count > 2:
            recs.append(f"You repeated words {repetition_count} times. Pause and gather your thoughts to avoid repetitions.")

        # Pitch
        pitch_variation_score = audio_features.get('pitch_variation_score', 0)
        if pitch_variation_score < 30:
            recs.append("Your pitch is monotone. Try to vary your tone to keep your audience engaged. Emphasize key words.")
        elif pitch_variation_score > 80:
            recs.append("Your pitch variation is high, which is great for engagement! Keep it up.")

        # Pace
        speaking_pace = audio_features.get('speaking_pace', 150)
        if speaking_pace > 180:
            recs.append(f"Your pace is very fast ({int(speaking_pace)} WPM). Slow down to ensure your audience can follow your message.")
        elif speaking_pace < 120 and speaking_pace > 0:
            recs.append(f"Your pace is a bit slow ({int(speaking_pace)} WPM). Try speaking a bit faster to maintain energy and engagement.")

        # Pauses and Silence
        if audio_features.get('silence_ratio', 0) > 0.4:
            recs.append("You have long or frequent pauses. While strategic pauses are good, too many can disrupt the flow.")

        # Confidence and Emotion
        if confidence_score < 60:
            recs.append("Your confidence score is a bit low. Practice more to feel more comfortable and speak with authority.")
        
        if emotion == 'tense':
            recs.append("You sound tense. Take deep breaths before speaking and try to relax your body.")
        
        if audio_features.get('rms_std', 0) > 0.05:
            recs.append("Your volume is a bit shaky. Project your voice from your diaphragm for a more stable and confident sound.")

        return recs if recs else ["Excellent delivery! Your speech patterns are clear and confident. Keep up the great work!"]
    # --- END OF COPIED PRIVATE METHODS ---


def perform_analysis_job(file_id: str, file_path: str, transcript: str) -> dict:
    """
    The main job function executed by the RQ worker.
    """
    audio_analysis_service = AudioAnalysisService()
    
    logger.info(f"[WORKER] Starting analysis for file_id: {file_id}")
    
    try:
        # 1. Perform the heavy analysis
        # Note: The result is a dictionary that RQ will serialize/store in Redis
        analysis_result = audio_analysis_service.analyze_audio(
            file_path=file_path,
            transcript=transcript
        )
        
        # 2. Add file_id for client reference
        analysis_result["file_id"] = file_id
        analysis_result["file_name"] = file_id 
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"[WORKER] Job {file_id} failed: {e}", exc_info=True)
        # RQ automatically marks the job as failed and stores the traceback
        raise # Re-raise to ensure RQ marks it as failed

    finally:
        # 3. Cleanup the file *after* analysis (success or failure)
        try:
            # We assume FileUploadService in main.py saved the file to UPLOAD_DIR
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"[WORKER] Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"[WORKER] Failed to delete file {file_path}: {e}")
        

# --- RQ Worker Startup Script ---
if __name__ == '__main__':
    # Get Redis connection string from environment variable
    # Render provides this via the internal Redis URL
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379') 
    
    # Simple check for Redis connection before starting worker
    try:
        conn = redis.from_url(redis_url)
        conn.ping()
        print(f"[WORKER] Connected to Redis at: {redis_url}. Starting worker...")
    except Exception as e:
        print(f"[WORKER] FATAL: Could not connect to Redis: {e}")
        exit(1) # Exit if Redis is unreachable

    with Connection(conn):
        # The 'default' queue is the one used by main.py
        worker = Worker(['default'], connection=conn)
        worker.work()
