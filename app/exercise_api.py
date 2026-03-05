from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from enum import Enum
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


@app.post("/exercises/generate", response_model=Exercise)
async def generate_exercise(user_id: str, custom_prompt: str):
    """Generates a custom exercise based on a user's free-form prompt."""
    print(f"Generating custom exercise for user_id: {user_id} with prompt: {custom_prompt}")
    user_data = {"user_id": user_id, "mode": "custom"}
    return await _call_ai_generator(custom_prompt, user_data, goal_type="custom")
