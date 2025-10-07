# backend/file_upload_service.py

import os
import uuid
import shutil
from fastapi import UploadFile
from datetime import datetime

# --- IMPORTANT: Define the upload directory ---
UPLOAD_DIR = "uploads" 
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def save_audio_file(file: UploadFile):
    """
    Saves the uploaded audio file and returns a dictionary with metadata.
    """
    
    # 1. Generate a unique ID (This is the critical 'file_id')
    file_id = str(uuid.uuid4())
    
    # 2. Define the file path
    # We use the file_id for a unique, safe name on the server
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{file_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    # 3. Save the file contents
    try:
        # Use a more robust asynchronous write for large files
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
            
        file_size = os.path.getsize(file_path)

    except Exception as e:
        # Clean up any partial file if saving failed
        if os.path.exists(file_path):
            os.remove(file_path)
        # Re-raise the exception to be caught in main.py
        raise e 
        
    # 4. CRITICAL: Return a dictionary that matches the UploadResponse model
    return {
        "status": "success",
        "file_id": file_id,
        "file_name": file.filename, 
        "file_size": file_size,
        "content_type": file.content_type,
        "upload_time": datetime.utcnow().isoformat()
    }