from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import Dict, Any, List
from datetime import datetime
import uuid
import os
import json
from openai import OpenAI # Import the OpenAI library

# --- Initialize OpenAI Client ---
# IMPORTANT: Replace 'YOUR_OPENAI_API_KEY' with a safe way to load your key,
# e.g., from an environment variable (recommended for production)
try:
    # Attempt to load the key from an environment variable
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"))
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    # Client will still be initialized, but calls will fail if the key is invalid

app = FastAPI(title="Intelligent Exercise System API", version="1.0.0")

# Pydantic Data Models (These are used to define the API contract)

class ExerciseType(str, Enum):
    PRONUNCIATION = "pronunciation"
    BREATHING = "breathing"
    PACE_CONTROL = "pace_control"
    CONFIDENCE = "confidence"
    PRESENTATION = "presentation"

class Exercise(BaseModel):
    # Enable model_json_schema() for Pydantic V2 to get structured schema
    model_config = ConfigDict(json_schema_extra={'example': {
        "type": "pace_control",
        "title": "The 60-Second Sales Pitch",
        "instructions": "Deliver a pitch for a pencil in exactly 60 seconds. Focus on even pacing.",
        "difficulty": 7,
        "duration_seconds": 60,
        "prompt": "Write a 60-second pitch for a simple object."
    }})
    
    exercise_id: str
    type: ExerciseType
    title: str
    instructions: str
    difficulty: int  # 1-10
    duration_seconds: int
    prompt: str

class PerformanceMetrics(BaseModel):
    score: float  # 0-100
    word_count: int
    errors_detected: Dict[str, int]
    pace_wpm: float
    breathing_efficiency: float

class ProgressUpdate(BaseModel):
    user_id: str
    exercise_id: str
    completed_at: datetime
    is_successful: bool
    audio_s3_key: str
    raw_transcript: str
    client_metrics: PerformanceMetrics

class RecommendationRequest(BaseModel):
    user_id: str
    last_completed_exercise_id: str
    current_weakness: str # e.g., "fast pace", "low confidence", "poor breathing"
    recent_scores: List[float]

class Achievement(BaseModel):
    achievement_name: str
    description: str
    unlocked_on: datetime
    tier: str

# --- Core LLM Logic Implementation ---

def _get_exercise_schema_for_openai() -> Dict[str, Any]:
    """Converts the Pydantic Exercise model schema for the OpenAI response_format."""
    # Use Pydantic's built-in schema generation (v2)
    schema = Exercise.model_json_schema()
    # We remove the exercise_id from the properties the LLM needs to generate, 
    # as we'll generate the UUID locally.
    if 'properties' in schema and 'exercise_id' in schema['properties']:
        del schema['properties']['exercise_id']
    if 'required' in schema and 'exercise_id' in schema['required']:
        schema['required'].remove('exercise_id')
        
    return schema

async def _call_ai_generator(prompt: str, user_data: Dict[str, Any], goal_type: str = "custom") -> Exercise:
    """Uses OpenAI API to generate a structured Exercise object."""
    exercise_schema = _get_exercise_schema_for_openai()

    if goal_type == "recommend":
        system_instruction = (
            "You are an expert speech coach AI. Based on the user's weakness, "
            "generate a single, highly-targeted practice exercise. "
            "The output MUST be a JSON object that strictly adheres to the provided schema."
        )
    else: # custom or default
        system_instruction = (
            "You are an expert speech coach AI, specializing in creating custom practice scenarios. "
            "Generate a structured speaking exercise based on the user's request. "
            "The output MUST be a JSON object that strictly adheres to the provided schema."
        )

    full_prompt = (
        f"Generate a speaking exercise. Prompt details: '{prompt}'. "
        f"User data for context (weakness, history, etc.): {user_data}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Using gpt-4o-mini for fast, cost-effective structured output
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "system", "content": f"JSON Schema: {json.dumps(exercise_schema)}"},
                {"role": "user", "content": full_prompt},
            ],
            response_format={"type": "json_object"}
        )

        json_response = response.choices[0].message.content
        exercise_data = json.loads(json_response)
        
        # Add the locally generated UUID before validating with Pydantic
        exercise_data['exercise_id'] = str(uuid.uuid4())
        
        # Validate and return the Pydantic model
        return Exercise(**exercise_data)
    
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        # Raise an HTTPException to stop the request and inform the client
        raise HTTPException(status_code=500, detail=f"AI generation failed. Check API key and logs: {e}")

# --- Core Logic Stubs (Still need Firestore implementation) ---

def _get_user_history(user_id: str) -> List[ProgressUpdate]:
    print(f"STUB: Retrieving user history for user_id: {user_id} - IMPLEMENT FIRESTORE READ.")
    return []

def _update_adaptive_difficulty(user_id: str, exercise_id: str, score: float) -> int:
    print(f"STUB: Updating adaptive difficulty for user_id: {user_id}, exercise_id: {exercise_id}, score: {score} - IMPLEMENT FIRESTORE UPDATE.")
    return 5

# --- FastAPI Endpoints (Now using implemented LLM logic) ---

@app.post("/exercises/recommend", response_model=Exercise)
async def recommend_exercise(request: RecommendationRequest):
    """Recommends an exercise based on user weakness and history."""
    
    # Construct a detailed prompt for the AI based on the user's data
    recommendation_prompt = f"The user's current primary weakness is '{request.current_weakness}'. Their recent scores were {request.recent_scores}. Generate an exercise specifically designed to challenge and improve this weakness."
    
    user_data = {
        "user_id": request.user_id,
        "recent_scores": request.recent_scores,
        "last_exercise": request.last_completed_exercise_id
    }
    
    # Use the LLM to generate the targeted exercise
    return await _call_ai_generator(recommendation_prompt, user_data, goal_type="recommend")

@app.post("/exercises/progress", status_code=202)
async def submit_progress(update: ProgressUpdate):
    """Submits and records exercise completion progress."""
    print(f"Submitting progress for user_id: {update.user_id}, exercise_id: {update.exercise_id} - IMPLEMENT FIRESTORE SAVE.")
    # This is where the ProgressUpdate object would be saved to Firestore.
    _update_adaptive_difficulty(update.user_id, update.exercise_id, update.client_metrics.score)
    return {"message": "Progress submitted and difficulty calculation triggered."}

@app.post("/exercises/analyze")
async def analyze_exercise(update: ProgressUpdate) -> Dict[str, Any]:
    """Uses LLM to generate detailed, actionable feedback on performance."""
    
    # Define the structure the AI must follow for the analysis response
    analysis_schema = {
        "type": "object",
        "properties": {
            "feedback": {"type": "string", "description": "Detailed, actionable feedback and advice for improvement. Use positive and encouraging language."},
            "summary": {"type": "string", "description": "A one-sentence summary comparing performance to goals or history."},
        },
        "required": ["feedback", "summary"],
    }
    
    system_instruction = (
        "You are an intelligent speech analysis AI. You must provide actionable, compassionate, "
        "and specific feedback on a user's speaking performance. "
        "The output MUST be a JSON object that strictly adheres to the provided schema."
    )

    prompt = (
        f"Analyze the following speaking exercise performance.\n"
        f"Raw Transcript: '{update.raw_transcript}'\n"
        f"Exercise ID: {update.exercise_id}\n"
        f"Metrics:\n"
        f"- Score: {update.client_metrics.score}/100\n"
        f"- Word Count: {update.client_metrics.word_count}\n"
        f"- Pace (WPM): {update.client_metrics.pace_wpm}\n"
        f"- Breathing Efficiency: {update.client_metrics.breathing_efficiency}\n"
        f"- Errors Detected: {update.client_metrics.errors_detected}\n"
        f"Provide detailed analysis focusing on how the user can improve their key weakness, based on the provided metrics. For example, if pace is low, suggest specific pacing techniques. If the score is low, highlight the top 2-3 areas for immediate focus."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "system", "content": f"JSON Schema: {json.dumps(analysis_schema)}"},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}
        )
        
        json_response = response.choices[0].message.content
        return json.loads(json_response)
        
    except Exception as e:
        print(f"OpenAI Analysis API call failed: {e}")
        raise HTTPException(status_code=500, detail="AI analysis failed. Please check backend logs.")


@app.get("/achievements/calculate", response_model=List[Achievement])
async def calculate_achievements(user_id: str):
    """Calculates achievements based on user history."""
    print(f"Calculating achievements for user_id: {user_id} - IMPLEMENT ACHIEVEMENT LOGIC.")
    # Stub logic for achievements (needs real logic based on user history)
    history = _get_user_history(user_id)
    # Simulate an achievement unlock for demo purposes
    return [
        Achievement(
            achievement_name="First Exercise",
            description="Completed your first exercise",
            unlocked_on=datetime.now(),
            tier="Bronze"
        )
    ]

@app.post("/exercises/generate", response_model=Exercise)
async def generate_exercise(user_id: str, custom_prompt: str):
    """Generates a custom exercise based on a user's free-form prompt."""
    print(f"Generating custom exercise for user_id: {user_id} with prompt: {custom_prompt}")
    user_data = {"user_id": user_id, "mode": "custom"}
    return await _call_ai_generator(custom_prompt, user_data, goal_type="custom")
