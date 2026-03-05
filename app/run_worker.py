#!/usr/bin/env python
"""
Worker launcher that ensures proper module imports for RQ.
Place this file in: app/run_worker.py
"""

import os
# REMOVED Numba/Librosa ENV VARS as they are no longer dependencies

impor
        # Start the worker
        logger.info("🎬 Starting RQ worker...")
        worker = Worker([queue], connection=redis_conn)
        worker.work()
        
    except Exception as e:
        # 🟢 FIX: Added the missing closing parenthesis ')' here
        logger.error(f"❌ Worker launcher failed: {e}", exc_info=True)
        sys.exit(1)
