#!/usr/bin/env python
"""
Worker launcher that ensures proper module imports for RQ.
Place this file in: app/run_worker.py
"""

# ⚠️ CRITICAL: Set Numba environment variables FIRST, before ANY imports
# This MUST be done before importing os, sys, or any other module
import os
# REMOVED Numba/Librosa ENV VARS as they are no longer dependencies
# os.environ['NUMBA_DISABLE_JIT'] = '1'
# os.environ['NUMBA_DISABLE_CUDA'] = '1'  
# os.environ['NUMBA_DISABLE_OPENMP'] = '1'
# os.environ['NUMBA_BOUNDSCHECK'] = '0'
# os.environ['LIBROSA_USE_NATIVE_MPG123'] = '1'

import sys
import logging

# Add parent directory to path so 'app' module can be imported
# This assumes the file is at app/run_worker.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis import Redis
from rq import Worker, Queue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [WORKER-LAUNCHER] - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    logger.info(f"🚀 Launching RQ worker, connecting to Redis at: {redis_url}")
    
    try:
        # Test Redis connection
        redis_conn = Redis.from_url(redis_url)
        redis_conn.ping()
        logger.info("✅ Redis connection established")
        
        # Create queue
        queue = Queue('default', connection=redis_conn)
        logger.info(f"✅ Queue 'default' created/connected")
        
        # Import worker module to ensure it's loaded (smoke test)
        logger.info("📦 Importing worker module...")
        from app import analysis_worker
        logger.info("✅ Worker module imported successfully")
        
        # Verify the function exists
        if hasattr(analysis_worker, 'perform_analysis_job'):
            logger.info("✅ perform_analysis_job function found")
        else:
            logger.error("❌ perform_analysis_job function NOT FOUND in module")
            sys.exit(1)
        
        # Start the worker
        logger.info("🎬 Starting RQ worker...")
        worker = Worker([queue], connection=redis_conn)
        worker.work()
        
    except Exception as e:
        logger.error(f"❌ Worker launcher failed: {e}", exc_info=True
