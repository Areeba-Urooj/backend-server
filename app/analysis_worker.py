# analysis_worker.py

import os
import logging

import soundfile as sf
import json

# --- IMPORTS FOR OPENAI (MANUAL HTTP) ---
        
   
        # FIX: Use a more robust threshold or ensure it's not too sensitive
        rms_threshold = np.mean(rms_frames) * 0.2 # Adjusted from 0.1 to 0.2 for better robustness 

        # FIX for long_pause_count: Calculate the actual count of DISTINCT long pause events
        long_pause_duration_frames = int(0.5 / (hop_len / sr)) # 0.5 second pause minimum
        
        is_silence = rms_frames < rms_threshold
        long_pause_count = 0
        in_long_pause = False
        pause_start_frame = -1

        for i in range(len(is_silence)):
            if is_silence[i]:
                if not in_long_pause:
                    in_long_pause = True
                    pause_start_frame = i
            elif in_long_pause:
                # End of silence segment
                pause_duration_frames = i - pause_start_frame
                if pause_duration_frames >= long_pause_duration_frames:
                    long_pause_count += 1
                in_long_pause = False

        # Check for pause at the very end
        if in_long_pause and (len(is_silence) - pause_start_frame) >= long_pause_duration_frames:
             long_pause_count += 1
        # END FIX

        # Uses the new NumPy/SciPy function
        pitch_mean, pitch_std = calculate_pitch_stats(y, sr)
        logger.info(f"📊 Pitch stats: mean={pitch_mean:.1f}Hz, std={pitch_std:.1f}Hz")
        
        audio_features_for_score = {
            "rms_mean": float(rms),
            "rms_std": float(energy_std), 
            "speaking_pace_wpm": speaking_pace_wpm,
            "pitch_std": float(pitch_std),
            "pitch_mean": float(pitch_mean),
        }
        
        # Uses the new NumPy/SciPy function
        acoustic_disfluencies = detect_acoustic_disfluencies(y, sr)
        serializable_disfluencies = [d._asdict() for d in acoustic_disfluencies]
        
        # Recalculate fluency metrics for scoring
        fluency_metrics_for_score = {
            "filler_word_count": filler_word_count,
            "repetition_count": repetition_count,
            "acoustic_disfluency_count": len(serializable_disfluencies),
            "total_words": total_words,
        }
        
        confidence_score = score_confidence(audio_features_for_score, fluency_metrics_for_score)
        
        # Emotion Classification
        emotion = classify_emotion_simple(temp_wav_file, EMOTION_MODEL, EMOTION_SCALER)
        
        # 7. Compile Core Metrics for LLM (UPDATED)
        core_analysis_metrics = {
            # Core metrics
            "confidence_score": round(confidence_score, 2),
            "speaking_pace": int(round(speaking_pace_wpm)),  # ✅ MUST be here
            "total_words": total_words,  # ✅ MUST be here
            "duration_seconds": round(duration_seconds, 2),

            # Fluency metrics
            "filler_word_count": filler_word_count,  # ✅ MUST be here
            "repetition_count": repetition_count,  # ✅ MUST be here
            "apology_count": apology_count,

            # Acoustic metrics
            "long_pause_count": int(long_pause_count),  # ✅ MUST be int
            "silence_ratio": round(silence_ratio, 4),

            # Audio features
            "pitch_mean": round(float(pitch_mean), 2),
            "pitch_std": round(float(pitch_std), 2),
            "avg_amplitude": round(float(rms), 6),
            "energy_std": round(float(energy_std), 4),

            # Other
            "emotion": emotion.lower(),
            "acoustic_disfluency_count": len(serializable_disfluencies),
            "transcript": transcript,

            # 🔥 CRITICAL: Include transcript_markers for highlighting
            "transcript_markers": [m._asdict() for m in all_text_markers],
        }

        # 8. Generate Intelligent Feedback
        logger.info("🤖 Generating intelligent feedback using OpenAI...")
        llm_recommendations = generate_intelligent_feedback(
            transcript=transcript,
            metrics=core_analysis_metrics
        )

      : {e}", exc_info=True)
        if 'redis' in str(e).lower():
            logger.critical("⚠️ Check your REDIS_URL and network configuration.")
        exit(1)

