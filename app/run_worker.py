#!/usr/bin/env python
"""
Worker launcher that ensures proper module imports for RQ.
This script sets up the Python path, connects to Redis, and starts the RQ worker
to process jobs from the 'default' queue.
"""

import os
import sys
import logging

# Essential: Set OMP_NUM_THREADS to 1 to prevent potential threading/resource issues
# in a multi-process worker environment.
os.environ['OMP_NUM_THREADS'] = '1'

# Add parent directory to the path so the 'app' module (containing analysis_worker) 
# can be imported correctly. Assuming the structure is:
# - project_root/
#   - app/
#     - run_worker.py (this file)
#     - analysis_worker.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis import Redis
from rq import Worker, Queue

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [WORKER-LAUNCHER] - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Get Redis URL from environment variable, defaulting to local Redis
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    logger.info(f"🚀 Launching RQ worker, connecting to Redis at: {redis_url}")
    
    try:
        # --- 1. Connection Test ---
        redis_conn = Redis.from_url(redis_url)
        redis_conn.ping()
        logger.info("✅ Redis connection established")
        
        # --- 2. Queue Setup ---
        # The worker will listen to this queue.
        queue = Queue('default', connection=redis_conn)
        logger.info(f"✅ Queue 'default' created/connected")
        
        # --- 3. Worker Module Smoke Test ---
        # This ensures the worker module is importable and the target function exists 
        # before the worker begins its work loop.
        logger.info("📦 Importing worker module...")
        from app import analysis_worker
        logger.info("✅ Worker module imported successfully")
        
        # Verify the function that jobs will call exists
        if hasattr(analysis_worker, 'perform_analysis_job'):
            logger.info("✅ perform_analysis_job function found")
        else:
            logger.error("❌ perform_analysis_job function NOT FOUND in module 'app.analysis_worker'")
            sys.exit(1)
            
        # --- 4. Start the Worker ---
        logger.info("🎬 Starting RQ worker...")
        # The Worker is initialized with the list of queues to listen to.
        worker = Worker([queue], connection=redis_conn)
        worker.work()
            
    except Exception as e:
        logger.error(f"❌ Worker launcher failed: {e}", exc_info=True)
        sys.exit(1)
