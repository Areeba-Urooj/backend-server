# app-folder/file_upload_service.py

import os
import uuid
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile
from datetime import datetime
from loguru import logger # 💡 NEW: Import logger for better error handling
from . import models

# --- AWS S3 Configuration ---
# 💡 NEW: Get S3 config from environment variables
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

# ❌ REMOVED: Local directory management is no longer needed
# UPLOAD_DIR = "uploads" 
# os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_s3_client():
    """Initializes and returns the S3 client."""
    if not S3_BUCKET_NAME:
        logger.error("[S3_SERVICE] ❌ S3_BUCKET_NAME environment variable is not set.")
        raise EnvironmentError("S3_BUCKET_NAME is not configured.")
    
    # Boto3 automatically uses AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
    return boto3.client('s3', region_name=AWS_REGION)


async def save_audio_file(file: UploadFile) -> models.UploadResponse:
    """
    Uploads the audio file directly to AWS S3 and returns an UploadResponse with the S3 Key.
    """
    s3_client = get_s3_client()
    
    # 1. Generate a unique ID (This is the critical 'S3 Key')
    # Use UUID to ensure keys don't clash, and keep the original file extension
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    # 💡 NEW: The S3 Key defines the path inside the bucket, e.g., 'uploads/a7b1c2d3.m4a'
    s3_key = f"uploads/{file_id}{file_extension}" 
    
    # 2. UPLOAD the file object directly to S3
    try:
        # file.file is a file-like object that boto3 can read from
        # file.read() is awaited in your original code, which is fine, but upload_fileobj is better
        # as it uses the file-like object directly without reading the whole thing into memory first
        
        # 💡 NOTE: We read the entire file into memory (content = await file.read()) to get the size later.
        # For very large files, stream it directly with upload_fileobj(Fileobj=file.file, ...)
        # We'll use your original method to simplify the transition and get the size.
        content = await file.read()
        file_size = len(content)
        
        # Rewind the file stream for S3 (important if it was read previously)
        # We use io.BytesIO to turn the content (bytes) into a file-like object for upload_fileobj
        import io
        file_stream = io.BytesIO(content) 

        s3_client.upload_fileobj(
            Fileobj=file_stream,
            Bucket=S3_BUCKET_NAME,
            Key=s3_key
        )
        
    except ClientError as e:
        logger.error(f"[S3_SERVICE] ❌ S3 Upload failed for key {s3_key}: {e}")
        # Re-raise the exception to be caught in main.py
        raise e
    except Exception as e:
        logger.error(f"[S3_SERVICE] ❌ Unknown error during upload: {e}")
        # Re-raise the exception to be caught in main.py
        raise e 
        
    # 3. Return the response model
    return models.UploadResponse(
        status="success",
        # 💡 CRITICAL: The file_id is now the S3 Key
        file_id=s3_key, 
        file_name=file.filename, 
        file_size=file_size,
        content_type=file.content_type or "application/octet-stream",
        upload_time=datetime.utcnow().isoformat()
    )

