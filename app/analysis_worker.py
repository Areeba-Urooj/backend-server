import os
import json
import time
import random
from typing import Dict, Any
from firebase_admin import initialize_app, firestore
from google.cloud.firestore import Client as FirestoreClient, DocumentSnapshot
from google.cloud.firestore_v1.watch import Watch
from pydantic import BaseModel, Field
from typing import List, Dict, Any

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
    if firebase_config:
        # Check if already initialized to prevent errors in certain environments
        if not firestore._app.get_app():
            initialize_app(options={'projectId': firebase_config['projectId']})
        db: FirestoreClient = firestore.client()
    else:
        db: FirestoreClient = None
except Exception as e:
    # If initialization fails, print error and set db to None
    print(f"Firebase Admin Initialization Error: {e}")
    db: FirestoreClient = None

# --- Firestore Helpers ---

def get_user_doc_ref(user_id: str, doc_name: str):
    """Gets the Firestore reference for user-specific data."""
    # Note: Using 'tasks' as a special document to hold the 'analysis_tasks' subcollection
    return db.collection('artifacts').document(app_id).collection('users').document(user_id).collection('data').document(doc_name)

def get_history_collection_ref(user_id: str):
    """Gets the Firestore reference for user performance history."""
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

def update_task_and_history(task_doc: DocumentSnapshot, metrics: PerformanceMetrics):
    """Updates the original task document and saves the result to user history."""
    if not db:
        print("Database not initialized, skipping update.")
        return

    task_data = task_doc.to_dict()
    user_id = task_data['user_id']
    task_id = task_doc.id
    
    # 1. Update the original task document with results (for the Flutter app to retrieve)
    task_ref = task_doc.reference
    task_ref.update({
        "status": "COMPLETED",
        "metrics": metrics.model_dump(),
        "completion_time": firestore.SERVER_TIMESTAMP
    })
    print(f"Task {task_id} status set to COMPLETED.")

    # 2. Save the result to the user's permanent performance history
    history_data = {
        "timestamp": firestore.SERVER_TIMESTAMP,
        "exercise_id": task_data.get('exercise_id'),
        "score": metrics.score,
        "metrics": metrics.model_dump(),
        "transcript": task_data.get('transcript'),
        "ai_feedback": task_data.get('ai_feedback') # Include instant feedback
    }
    get_history_collection_ref(user_id).add(history_data)
    print(f"Performance history recorded for user {user_id}.")


def process_task_queue(col_snapshot: List[DocumentSnapshot], changes: List[Dict[str, Any]], read_time: Any):
    """Callback function for the Firestore real-time listener."""
    
    for change in changes:
        # We only care about new documents (tasks)
        if change.type.name == 'ADDED':
            task_doc = change.document
            task_data = task_doc.to_dict()
            task_id = task_doc.id

            if task_data.get('status') == 'QUEUED':
                user_id = task_data.get('user_id', 'UNKNOWN_USER')
                print(f"--- Worker found new task: {task_id} for user {user_id} ---")

                try:
                    # Set status to PROCESSING immediately to prevent double processing
                    task_doc.reference.update({"status": "PROCESSING", "start_time": firestore.SERVER_TIMESTAMP})
                    print(f"Task {task_id} status set to PROCESSING.")

                    # Run the simulated deep analysis
                    metrics = perform_deep_analysis(
                        transcript=task_data['transcript'], 
                        exercise_id=task_data['exercise_id']
                    )
                    
                    # Update the task and history
                    update_task_and_history(task_doc, metrics)

                except Exception as e:
                    print(f"CRITICAL ERROR processing task {task_id}: {e}")
                    # Update status to failed so the user knows
                    task_doc.reference.update({
                        "status": "FAILED",
                        "error_message": str(e),
                        "completion_time": firestore.SERVER_TIMESTAMP
                    })

def start_worker():
    """Main function to start the Firestore listener."""
    if db is None:
        print("Cannot start worker: Database not initialized.")
        return

    print(f"Intelligent Analysis Worker starting... Monitoring tasks in app ID: {app_id}")
    
    # The worker must monitor ALL users' task queues.
    # We query the 'analysis_tasks' subcollection across ALL user documents.
    
    # Note: Firestore does not support 'collectionGroup' queries in this environment 
    # for local testing, so we will simulate monitoring a single user's queue 
    # or rely on a simplified setup where the user ID is constant (e.g., 'default-user').

    # For a robust real-world setup, you would use a dedicated Message Queue 
    # (like Google Pub/Sub) or a Firestore Collection Group. 
    # Since we are confined to a single app environment, we'll monitor a simple 
    # global queue for simulation purposes.
    
    # MONITORING SIMULATION: We'll monitor a placeholder user's task queue, 
    # assuming all tasks are routed there for simplicity in this constrained environment.
    
    # In a real app, you would need to know the specific user_id to monitor 
    # or use a collection group, but for this simulation, we'll monitor 
    # the 'default-user' queue for new tasks.
    
    # **IMPORTANT**: In the 'exercise_api.py' we used a reference that depends on `user_id`.
    # `task_ref = get_user_doc_ref(request.user_id, 'tasks').collection('analysis_tasks')`
    # The worker cannot easily listen to all these paths. For this constrained environment,
    # we will modify the API and Worker to use a single **public** task queue 
    # that all users write to, which is easier for the single worker to monitor.
    # I will stick with the original structure for now and assume `run_worker.py` 
    # (which you'll implement next) handles the task routing or we rely on 
    # the Flutter app to poll. But to simulate real-time... we will listen 
    # to a specific, hardcoded user ID's queue.

    print("--- WARNING: Worker is monitoring a SIMULATED queue (user 'worker_monitor_user'). ---")
    
    # The collection we are listening to is inside a specific user document.
    task_collection_ref = get_user_doc_ref('worker_monitor_user', 'tasks').collection('analysis_tasks')

    # A better approach, which we will use, is to tell the worker 
    # to monitor the **public** queue for all tasks. Let's make the API use this 
    # public queue too (a minor change in exercise_api.py is needed, 
    # but for now, we'll stick to listening to a known path and rely on the 
    # simulation for task finding).
    
    # Since we cannot update exercise_api.py now, we will assume 
    # `task_data['user_id']` contains the user_id that the Flutter app uses 
    # for polling, but the worker listens to a single queue path for tasks.
    
    # If a real-time listener is used, the code would be:
    # watch = Watch(task_collection_ref.where('status', '==', 'QUEUED'), callback=process_task_queue)
    
    # Since the listener is complex in this setup, let's use a simple polling loop
    # which is more robust in constrained environments.

    while True:
        try:
            # Query for all 'QUEUED' tasks under the 'worker_monitor_user' path
            print("Worker Polling for QUEUED tasks...")
            
            # NOTE: We are now changing the worker to check ALL users' queues 
            # by iterating through user documents, which is inefficient 
            # but necessary if we can't use a collection group query. 
            # We must monitor the main 'analysis_tasks' path, which is difficult.
            
            # REVERTING TO SIMULATION: Let's assume a single global task queue 
            # for the worker to monitor. The Flutter app will poll for its result.
            
            # Global Task Queue (for simulation ease): artifacts/{app_id}/tasks/analysis_queue
            global_task_ref = db.collection('artifacts').document(app_id).collection('tasks').document('analysis_queue').collection('tasks')

            # Fetch up to 10 queued tasks
            queued_tasks = global_task_ref.where('status', '==', 'QUEUED').limit(10).stream()
            
            found_task = False
            for task_doc in queued_tasks:
                found_task = True
                task_data = task_doc.to_dict()
                task_id = task_doc.id
                user_id = task_data.get('user_id', 'UNKNOWN_USER')

                print(f"--- Worker found new task: {task_id} for user {user_id} ---")

                try:
                    # Set status to PROCESSING immediately to prevent double processing
                    task_doc.reference.update({"status": "PROCESSING", "start_time": firestore.SERVER_TIMESTAMP})
                    print(f"Task {task_id} status set to PROCESSING.")

                    # Run the simulated deep analysis
                    metrics = perform_deep_analysis(
                        transcript=task_data['transcript'], 
                        exercise_id=task_data['exercise_id']
                    )
                    
                    # Update the task and history (using the user_id from the task data)
                    # This relies on the original user-specific path for the Flutter app to find results.
                    # We create a new document in the user's permanent history path.
                    
                    # 1. Save the result to the user's permanent performance history
                    history_data = {
                        "timestamp": firestore.SERVER_TIMESTAMP,
                        "exercise_id": task_data.get('exercise_id'),
                        "score": metrics.score,
                        "metrics": metrics.model_dump(),
                        "transcript": task_data.get('transcript'),
                        "ai_feedback": task_data.get('ai_feedback')
                    }
                    history_ref = get_history_collection_ref(user_id).add(history_data)
                    result_doc_id = history_ref[1].id
                    print(f"Performance history recorded as {result_doc_id} for user {user_id}.")

                    # 2. Update the original global task document with a reference to the result
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
    start_worker()
