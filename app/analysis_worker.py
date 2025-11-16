import os
import json
import time
import random
from typing import Dict, Any, List
from firebase_admin import initialize_app, firestore
from google.cloud.firestore import Client as FirestoreClient, DocumentSnapshot
from pydantic import BaseModel, Field

# --- Environment Setup ---
# __app_id is used for namespacing Firestore collections
app_id = os.environ.get('__app_id', 'default-speech-app')
# __firebase_config is used to initialize the Firebase Admin SDK
firebase_config = json.loads(os.environ.get('__firebase_config', '{}'))

# --- Pydantic Models for Data Structure ---

class PerformanceMetrics(BaseModel):
    """The complete set of metrics calculated by the analysis worker."""
    score: int = Field(..., ge=0, le=100, description="Overall quality score (0-100).")
    pace_wpm: int = Field(..., description="Words Per Minute.")
    breathing_efficiency: float = Field(..., ge=0.0, le=1.0, description="Efficiency score (0.0 to 1.0).")
    errors_detected: list[str] = Field(..., description="Specific issues like 'Stutter', 'Long Pause', 'Mumble'.")
    adaptive_difficulty_change: str = Field(..., description="The next recommended difficulty level.")
    
# --- Firebase Initialization and Services ---

try:
    db: FirestoreClient = None
    if firebase_config:
        # Check if already initialized to prevent errors in certain environments
        # Note: firestore._app.get_app() might not be available in all envs, relying on standard check
        try:
            firestore.get_app()
        except ValueError:
            initialize_app(options={'projectId': firebase_config['projectId']})
        db = firestore.client()
except Exception as e:
    # If initialization fails, print error and set db to None
    print(f"Firebase Admin Initialization Error: {e}")
    db = None

# --- Firestore Helpers ---

def get_user_doc_ref(user_id: str, doc_name: str):
    """Gets the Firestore reference for user-specific data."""
    # Note: Using 'tasks' as a special document to hold the 'analysis_tasks' subcollection
    if not db: raise ConnectionError("Database not initialized.")
    return db.collection('artifacts').document(app_id).collection('users').document(user_id).collection('data').document(doc_name)

def get_history_collection_ref(user_id: str):
    """Gets the Firestore reference for user performance history."""
    if not db: raise ConnectionError("Database not initialized.")
    return db.collection('artifacts').document(app_id).collection('users').document(user_id).collection('performance_history')

# --- Core Worker Logic ---

def perform_deep_analysis(transcript: str, exercise_id: str) -> PerformanceMetrics:
    """
    SIMULATION: This function simulates the heavy lifting of speech analysis.
    In a real system, this would involve calling a complex ML/NLP model 
    (like analysis_engine.py).
    """
    print(f"--> Analyzing Exercise {exercise_id} (Transcript length: {len(transcript)})...")
    time.sleep(random.uniform(3, 8)) # Simulate processing time

    # 1. Simulated Score Calculation
    base_score = random.randint(70, 95)
    
    # 2. Simulated Metrics
    words = len(transcript.split())
    duration_sec = random.randint(45, 120) 
    pace = round(words / (duration_sec / 60))
    
    # 3. Simulated Error Detection (Tied to weaknesses)
    potential_errors = ["Stutter", "Long Pause", "Mumble", "Filler Word Usage", "Low Projection"]
    errors = random.sample(potential_errors, k=random.randint(0, 3))
    
    # 4. Simulated Adaptive Difficulty Logic
    new_difficulty = random.choice(["Beginner", "Intermediate", "Advanced"])

    metrics = PerformanceMetrics(
        score=base_score,
        pace_wpm=pace,
        breathing_efficiency=random.uniform(0.5, 0.95),
        errors_detected=errors,
        adaptive_difficulty_change=new_difficulty
    )
    
    print(f"<-- Analysis Complete. Score: {metrics.score}, Next Difficulty: {metrics.adaptive_difficulty_change}")
    return metrics

def perform_analysis_job():
    """
    Worker entry point: Polls a global task queue for new analysis requests.
    This function satisfies the name required by the external worker launcher.
    """
    if db is None:
        print("Cannot start worker: Database not initialized.")
        return

    print(f"Intelligent Analysis Worker starting... Monitoring tasks in app ID: {app_id}")
    
    # Global Task Queue (for simulation ease): artifacts/{app_id}/tasks/analysis_queue/tasks
    # The worker polls this queue for new tasks from all users.
    
    while True:
        try:
            global_task_ref = db.collection('artifacts').document(app_id).collection('tasks').document('analysis_queue').collection('tasks')

            # Fetch up to 10 queued tasks
            print("Worker Polling for QUEUED tasks...")
            
            # NOTE: We use .stream() instead of .get() for consistency
            queued_tasks = global_task_ref.where('status', '==', 'QUEUED').limit(10).stream()
            
            found_task = False
            for task_doc in queued_tasks:
                found_task = True
                task_data = task_doc.to_dict()
                task_id = task_doc.id
                user_id = task_data.get('user_id', 'UNKNOWN_USER')

                print(f"--- Worker found new task: {task_id} for user {user_id} ---")

                try:
                    # 1. Set status to PROCESSING immediately to prevent double processing
                    task_doc.reference.update({"status": "PROCESSING", "start_time": firestore.SERVER_TIMESTAMP})
                    print(f"Task {task_id} status set to PROCESSING.")

                    # 2. Run the simulated deep analysis
                    metrics = perform_deep_analysis(
                        transcript=task_data['transcript'], 
                        exercise_id=task_data['exercise_id']
                    )
                    
                    # 3. Save the result to the user's permanent performance history
                    history_data = {
                        "timestamp": firestore.SERVER_TIMESTAMP,
                        "exercise_id": task_data.get('exercise_id'),
                        "score": metrics.score,
                        "metrics": metrics.model_dump(),
                        "transcript": task_data.get('transcript'),
                        "ai_feedback": task_data.get('ai_feedback')
                    }
                    # We add to the user's private history collection
                    history_ref = get_history_collection_ref(user_id).add(history_data)
                    result_doc_id = history_ref[1].id
                    print(f"Performance history recorded as {result_doc_id} for user {user_id}.")

                    # 4. Update the original global task document as COMPLETED
                    task_doc.reference.update({
                        "status": "COMPLETED",
                        "metrics_result_id": result_doc_id, # ID where the Flutter app can find the detailed results
                        "completion_time": firestore.SERVER_TIMESTAMP
                    })
                    print(f"Task {task_id} status set to COMPLETED and links to result ID: {result_doc_id}.")

                except Exception as e:
                    print(f"CRITICAL ERROR processing task {task_id}: {e}")
                    task_doc.reference.update({
                        "status": "FAILED",
                        "error_message": str(e),
                        "completion_time": firestore.SERVER_TIMESTAMP
                    })
            
            if not found_task:
                print("No new tasks found. Waiting...")

        except Exception as e:
            print(f"Worker main loop error: {e}. Retrying in 10 seconds.")
        
        # Wait before polling again
        time.sleep(10)


if __name__ == "__main__":
    perform_analysis_job()
