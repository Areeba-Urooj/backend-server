import os
import uuid
import logging
import re
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

import librosa
import numpy as np
import scipy.signal as signal
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends
from pydantic import BaseModel, Field

# --- Configuration ---
# In a real app, this would be in a .env file or other config management
UPLOAD_DIR = "uploads_simple"
ALLOWED_AUDIO_TYPES = [
    ".wav", ".mp3", ".m4a", ".aac"
]
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

# --- FastAPI App ---
app = FastAPI(
    title="PodiumPal Simple Analysis API",
    description="A simplified, single-file version of the speech analysis backend.",
    version="1.0.0"
)

# --- Pydantic Schemas (Request/Response Models) ---

class AnalysisRequest(BaseModel):
    file_id: str = Field(..., description="ID of the uploaded audio file to analyze")
    transcript: Optional[str] = Field(None, description="Optional transcript text if already available")

class AudioFeatures(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    rms_mean: float
    rms_std: float
    pitch_mean: float
    pitch_std: float
    pitch_min: float
    pitch_max: float
    speaking_pace: float
    silence_ratio: float
    zcr_mean: float

class FillerWordAnalysis(BaseModel):
    filler_word_count: int
    filler_words: Dict[str, int]
    filler_word_ratio: float

class SpeechAnalysisResponse(BaseModel):
    file_id: str
    duration_seconds: float
    audio_features: AudioFeatures
    filler_word_analysis: FillerWordAnalysis
    repetition_count: int
    long_pause_count: int
    total_words: int
    confidence_score: float
    emotion: str
    pitch_variation_score: float
    recommendations: List[str]
    analyzed_at: datetime

class UploadResponse(BaseModel):
    file_id: str
    file_name: str
    file_size: int
    content_type: str
    upload_time: datetime
    status: str = "success"

# --- Audio Analysis Service Logic ---

class AudioAnalysisService:
    def __init__(self):
        self.filler_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in FILLER_WORDS) + r')\b',
            re.IGNORECASE
        )

    async def analyze_audio(self, file_path: str, transcript: Optional[str] = None) -> Dict[str, Any]:
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
                transcript_analysis.get('filler_word_analysis', {})
            )
            emotion = self._determine_emotion(audio_features)
            recommendations = self._generate_recommendations(
                audio_features,
                transcript_analysis.get('filler_word_analysis', {}),
                confidence_score,
                emotion
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
                "analyzed_at": datetime.now()
            }

        except Exception as e:
            logger.error(f"Error analyzing audio: {e}", exc_info=True)
            raise

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
        score -= min(30, filler_analysis.get('filler_word_ratio', 0) * 150) # Increased penalty
        
        # Penalty for long silences
        if audio_features.get('silence_ratio', 0) > 0.3:
            score -= min(20, (audio_features.get('silence_ratio', 0) - 0.3) * 100)
            
        # Bonus for good pitch variation, penalty for monotone
        pitch_variation_score = audio_features.get('pitch_variation_score', 0)
        score += min(15, (pitch_variation_score - 50) / 5) if pitch_variation_score > 50 else -min(20, (50 - pitch_variation_score) / 4) # Increased penalty for monotone
        
        # Penalty for speaking too fast or too slow
        pace_deviation = abs(audio_features.get('speaking_pace', 150) - 150) / 150
        score -= min(20, pace_deviation * 60) # Increased penalty
        
        # Penalty for unstable energy (shaky voice)
        rms_std = audio_features.get('rms_std', 0)
        if rms_std > 0.05: # Heuristic threshold for shakiness
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

# --- File Upload Service Logic ---

class FileUploadService:
    def __init__(self):
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    async def save_audio_file(self, file: UploadFile) -> Dict[str, Any]:
        logger.info(f"Uploading file: {file.filename}, content type: {file.content_type}, size: {file.size}")
        file_extension = os.path.splitext(file.filename)[1].lower()
        logger.info(f"File extension: {file_extension}")
        if file_extension not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed: {ALLOWED_AUDIO_TYPES}")

        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        saved_size = os.path.getsize(file_path)
        logger.info(f"Saved file size: {saved_size}")

        return {
            "file_id": file_id,
            "file_name": file.filename,
            "file_size": saved_size,
            "content_type": file.content_type,
            "upload_time": datetime.now()
        }

    def get_file_path(self, file_id: str) -> str:
        for ext in ALLOWED_AUDIO_TYPES:
            file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
            if os.path.exists(file_path):
                return file_path
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")

# --- API Endpoints ---

audio_analysis_service = AudioAnalysisService()
file_upload_service = FileUploadService()

@app.get("/")
async def read_root():
    return {"message": "PodiumPal Simple Analysis API is running!"}

@app.on_event("startup")
async def startup_event():
    # Clean up old uploads on startup
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

@app.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    try:
        result = await file_upload_service.save_audio_file(file)
        return result
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=SpeechAnalysisResponse)
async def analyze_speech(request: AnalysisRequest):
    try:
        file_path = file_upload_service.get_file_path(request.file_id)
        analysis_result = await audio_analysis_service.analyze_audio(
            file_path=file_path,
            transcript=request.transcript
        )
        analysis_result["file_id"] = request.file_id
        # The file name is not available here without storing it.
        # For simplicity, we'll return the file_id as the name.
        analysis_result["file_name"] = request.file_id
        return analysis_result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error analyzing speech: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)