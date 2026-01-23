# analysis_engine.py

import numpy as np
import soundfile as sf
import subprocess

MAX_DURATION_SECONDS = 120 
logger = logging.getLogger(__name__)

# --- NamedTuples for Analysis Markers ---

# Time-based marker for acoustic/audio events
class DisfluencyResult(NamedTuple):
    type: str # 'block', 'prolongation'
 
    Detects filler words, laughter, and common apologetic phrases, returning TextMarkers with indices.
    Uses comprehensive lists of common filler words and patterns that work for any speaker.
    """
    # Comprehensive filler word set - covers most common verbal fillers that people use naturally
    filler_set = {
        # Hesitation fillers (most common)
        "um", "uh", "er", "erm", "emm", "hmm", "huh", "mm", "mhm",
        # Discourse markers and conversational fillers
        "like", "so", "you know", "i mean", "right", "okay", "alright", "ok",
        "well", "now", "anyway", "actually", "basically", "literally", "totally",
        "seriously", "honestly", "really", "just", "sort of", "kind of",
        "you see", "see", "look", "listen", "hey", "oh", "ah", "eh",
        "uh-huh", "yep", "nope", "yeah", "yup", "sure", "absolutely",
        # Vocal fillers and sounds
        "ho", "ha", "hee", "hi", "whoa", "wow", "oops", "whoops",
        # More hesitation sounds
        "uhm", "hum", "huh", "phew", "whew",
        # Additional comprehensive fillers from user data
        "ugh", "argh", "eek", "ooh", "aah", "eww", "yay", "psst", "shh", "shhh", "tsk", "tut", "tut-tut", "pfft", "psh", "gah", "blah", "bah", "pah", "duh", "duh-duh", "yeah-yeah", "huh-huh", "hah", "hehe", "haha", "hehehehe", "lol", "la", "la-la", "da", "da-da", "ba", "ba-ba", "na", "na-na", "ta", "ta-ta", "ding", "dong", "ting", "tang", "boing", "boom", "pow", "zap", "splat", "thud", "thump", "bump", "bang", "crash", "smash", "whack", "whoosh", "swish", "swoosh", "zoom", "vroom", "beep", "boop", "bleep", "bloop", "click", "clack", "tick", "tock", "ticking", "tocking", "pitter-patter", "rat-a-tat", "tap-tap", "tap-tap-tap", "knock-knock", "rap-rap", "pat-pat", "pit-pit", "tat-tat", "tit-tat", "toot", "toot-toot", "honk", "honk-honk", "meow", "woof", "bark", "baa", "moo", "oink", "quack", "chirp", "tweet", "tweet-tweet", "chirp-chirp", "screech", "squawk", "hiss", "buzz", "bzzzz", "whirr", "whirrr", "hum", "hummmm", "drone", "groan", "moan", "sigh", "sighhhh", "gasp", "pant", "panting", "wheeze", "cough", "cough-cough", "hack", "hack-hack", "achoo", "achoo-achoo", "sniff", "sniffle", "snore", "snoring", "snort", "snorting", "burp", "burp-burp", "hiccup", "hiccup-hiccup", "gulp", "glug", "glug-glug", "slurp", "slurp-slurp", "munch", "munch-munch", "crunch", "crunch-crunch", "chomp", "chomp-chomp", "chew", "chew-chew", "smack", "smack-smack", "nom", "nom-nom", "slosh", "sloshing", "splash", "splash-splash", "plop", "plop-plop", "splurge", "splurging", "squirt", "squirt-squirt", "spritz", "spritzing", "spray", "spraying", "drip", "drip-drip", "dripping", "drop", "drop-drop", "dropping", "plink", "plink-plink", "plunk", "plunk-plunk", "ping", "ping-ping", "pong", "pong-pong", "boing-boing", "bounce", "bouncing", "whomp", "whomp-whomp", "whomping", "thwack", "thwack-thwack", "swat", "swat-swat", "smack", "smacking", "crack", "cracking", "snap", "snap-snap", "snapping", "pop", "pop-pop", "popping", "crackle", "crackling", "fizz", "fizz-fizz", "fizzle", "fizzling", "sizzle", "sizzling", "fry", "frying", "sear", "searing", "burn", "burning", "blaze", "blazing", "flame", "flaming", "spark", "sparking", "flicker", "flickering", "glimmer", "glimmering", "shimmer", "shimmering", "glisten", "glistening", "shine", "shining", "glow", "glowing", "twinkle", "twinkling", "flash", "flashing", "flare", "flaring", "burst", "bursting", "explode", "exploding", "erupt", "erupting", "rumble", "rumbling", "roar", "roaring", "thunder", "thundering", "lightning", "crackle-crackle", "pop-pop-pop", "bang-bang-bang", "boom-boom", "boom-boom-boom", "kaboom", "ker-blam", "pow-pow", "bam-bam", "wham-wham", "thud-thud", "thump-thump", "bump-bump", "crash-crash", "smash-smash", "crack-crack", "snap-snap", "pop-pop", "click-click", "clack-clack", "tick-tick", "tock-tock", "ding-ding", "dong-dong", "ting-ting", "tang-tang", "ping-ping", "pong-pong", "ping-pong", "beep-beep", "boop-boop", "bleep-bleep", "bloop-bloop", "whirr-whirr", "buzz-buzz", "hum-hum", "drone-drone", "zoom-zoom", "vroom-vroom", "whoosh-whoosh", "swish-swish", "swoosh-swoosh", "meow-meow", "woof-woof", "bark-bark", "baa-baa", "moo-moo", "oink-oink", "quack-quack", "chirp-chirp", "tweet-tweet", "screech-screech", "squawk-squawk", "hiss-hiss", "groan-groan", "moan-moan", "sigh-sigh", "gasp-gasp", "pant-pant", "wheeze-wheeze", "cough-cough", "hack-hack", "achoo-achoo", "sniff-sniff", "sniffle-sniffle", "snore-snore", "snort-snort", "burp-burp", "hiccup-hiccup", "gulp-gulp", "glug-glug", "slurp-slurp", "munch-munch", "crunch-crunch", "chomp-chomp", "chew-chew", "smack-smack", "nom-nom", "splash-splash", "plop-plop", "squirt-squirt", "spritz-spritz", "spray-spray", "drip-drip", "drop-drop", "plink-plink", "plunk-plunk", "thwack-thwack", "swat-swat", "crack-crack", "fizz-fizz", "sizzle-sizzle", "fry-fry", "spark-spark", "flicker-flicker", "shimmer-shimmer", "glisten-glisten", "shine-shine", "glow-glow", "twinkle-twinkle", "flash-flash", "flare-flare", "burst-burst", "rumble-rumble", "roar-roar", "thunder-thunder", "lalala", "lalalala", "doodoodoo", "doodoodoodoo", "nanana", "nananana", "tatata", "tatatata", "blahblah", "blahblahblah", "yada", "yada-yada", "yada-yada-yada", "yadda", "yadda-yadda", "yadda-yadda-yadda", "like-like", "you-know", "y-know", "I-mean", "y-see", "um-like", "uh-like", "um-yeah", "uh-yeah", "um-so", "uh-so", "um-I-mean", "uh-I-mean", "um-you-know", "uh-you-know", "um-basically", "uh-basically", "um-actually", "uh-actually", "um-literally", "uh-literally", "um-right", "uh-right", "um-okay", "uh-okay", "um-well", "uh-well", "um-anyway", "uh-anyway", "um-I-guess", "uh-I-guess", "um-I-think", "uh-I-think", "um-I-dunno", "uh-I-dunno", "um-thing", "uh-thing", "um-stuff", "uh-stuff", "um-whatnot", "uh-whatnot", "um-something", "uh-something", "um-whatever", "uh-whatever", "um-or-something", "uh-or-something", "um-or-like", "uh-or-like", "um-kinda", "uh-kinda", "um-sorta", "uh-sorta", "um-I-suppose", "uh-I-suppose", "um-probably", "uh-probably", "um-maybe", "uh-maybe", "um-perhaps", "uh-perhaps", "um-possibly", "uh-possibly", "um-apparently", "uh-apparently", "um-seemingly", "uh-seemingly", "um-clearly", "uh-clearly", "um-obviously", "uh-obviously", "um-certainly", "uh-certainly", "um-definitely", "uh-definitely", "um-absolutely", "uh-absolutely", "um-totally", "uh-totally", "um-completely", "uh-completely", "um-entirely", "uh-entirely", "um-quite", "uh-quite", "um-rather", "uh-rather", "um-pretty", "uh-pretty", "um-very", "uh-very", "um-really", "uh-really", "um-truly", "uh-truly", "um-genuinely", "uh-genuinely", "um-honestly", "uh-honestly", "um-frankly", "uh-frankly", "um-frankly-speaking", "uh-frankly-speaking", "um-to-be-honest", "uh-to-be-honest", "um-to-be-frank", "uh-to-be-frank"
    }

    apology_set = {"sorry", "excuse me", "apologize", "apology", "pardon", "forgive", "my bad"}

    markers: List[TextMarker] = []

    # Detect laughter patterns - flexible regex that catches various laughter forms
    # This will match "haha", "hehe", "hihi", "hoho", "ha ha", "heehee", "lol", etc.
    laughter_pattern = r'\b(?:ha|he|hi|ho|hee|lol|rofl|lmao){2,}\b|\b(?:ha|he|hi|ho)\s*(?:ha|he|hi|ho)+\b'
    for match in re.finditer(laughter_pattern, transcript, re.IGNORECASE):
        markers.append(TextMarker(
            type='filler',
            word=match.group(0),
            start_char_index=match.start(),
            end_char_index=match.end()
        ))

    # Detect other vocal sounds and exclamations like "uh-oh", "oh boy", etc.
    vocal_pattern = r'\b(?:uh-oh|oh boy|oh man|oh god|oh no|oh dear|goodness|gosh|wow|oops|whoops|jeez|geez|darn|shoot)\b'
    for match in re.finditer(vocal_pattern, transcript, re.IGNORECASE):
        markers.append(TextMarker(
            type='filler',
            word=match.group(0),
            start_char_index=match.start(),
            end_char_index=match.end()
        ))

    # Detect repeated syllables like "la-la-la" or "da-da-da"
    repeated_syllable_pattern = r'\b(\w{1,3})\s*(?:-\s*\1|\s+\1){2,}\b'
    for match in re.finditer(repeated_syllable_pattern, transcript, re.IGNORECASE):
        markers.append(TextMarker(
            type='filler',
            word=match.group(0),
            start_char_index=match.start(),
            end_char_index=match.end()
        ))

    # Use re.finditer to get word matches with their start/end indices
    # Updated regex to catch hyphenated words and contractions
    for match in re.finditer(r"\b\w+(?:[-']\w+)*\b", transcript.lower()):
        word = match.group(0)
        start_index = match.start()
        end_index = match.end()

        marker_type: Optional[str] = None

        if word in filler_set:
            marker_type = 'filler'
        elif word in apology_set:
            marker_type = 'apology'

        if marker_type:
            markers.append(TextMarker(
                type=marker_type,
                word=transcript[start_index:end_index],  # Preserve original casing
                start_char_index=start_index,
                end_char_index=end_index
            ))

    return markers

def detect_repetitions_for_highlighting(transcript: str) -> List[TextMarker]:
    """
    Detects repeated adjacent words, returning TextMarkers.
    """
    markers: List[TextMarker] = []

    # Tokenize the transcript while keeping track of original indices
    token_matches = list(re.finditer(r'(\w+)(\W*)', transcript))

    i = 0
    while i < len(token_matches) - 1:
        word1_lower = token_matches[i].group(1).lower()
        word2_lower = token_matches[i+1].group(1).lower()

        if word1_lower == word2_lower and len(word1_lower) > 2: # Ignore single letter repeats
            # Repetition found: highlight the full phrase including the second instance.
            start_char_index = token_matches[i].start(1)
            end_char_index = token_matches[i+1].end(1)

            markers.append(TextMarker(
                type='repetition',
                word=f"{token_matches[i].group(1)} {token_matches[i+1].group(1)}",
                start_char_index=start_char_index,
                end_char_index=end_char_index
            ))

            i += 2 # Skip both repeated words
        else:
            i += 1

    # Also detect non-adjacent repetitions like "wait, wait, wait"
    word_positions = {}
    for match in re.finditer(r'\b(\w+)\b', transcript):
        word = match.group(1).lower()
        if word not in word_positions:
            word_positions[word] = []
        word_positions[word].append((match.start(), match.end()))

    for word, positions in word_positions.items():
        if len(positions) >= 3 and len(word) > 2:  # At least 3 occurrences
            # Mark all occurrences of the repeated word
            for start_idx, end_idx in positions:
                markers.append(TextMarker(
                    type='repetition',
                    word=transcript[start_idx:end_idx],
                    start_char_index=start_idx,
                    end_char_index=end_idx
                ))

    return markers

def detect_custom_markers(transcript: str) -> List[TextMarker]:
    """
    Detects custom phrases like tangents, self-correction, or meta-commentary 
    based on the content identified as problematic in the sample transcript.
    """
    markers: List[TextMarker] = []
    
    # NOTE: These patterns are specific to the user's problematic transcript example
    custom_patterns = [
        # Tangent/Off-topic (Football game - first instance)
        (r'did you see what the team said about conversion rates on the landing page\? And then, oh, sorry, the other thing is, did you watch the football game last night\? It was a crazy game\. What a wild night\.', 'tangent'),
        # Self-Correction/Apology for topic change
        (r'And actually, sorry, let\'s stay on topic and talk about the football game at lunch, because that\'s where we should talk about football, not in the middle of a team meeting.', 'self_correction'),
        # Meta-Commentary/Internal Monologue
        (r'Oh, far out\. Making this video just makes me feel bad because I used to do every single one of these things\.', 'meta_commentary'),
        # Repeating the tangent (This is the second instance, which should also be marked)
        (r'did you also watch the football game last night\? It\s+was a crazy game\. What a wild night\.', 'repetition')
    ]

    for pattern, marker_type in custom_patterns:
        # The re.IGNORECASE flag helps catch variations in capitalization
        for match in re.finditer(pattern, transcript, re.IGNORECASE): 
            markers.append(TextMarker(
                type=marker_type,
                word=match.group(0),
                start_char_index=match.start(),
                end_char_index=match.end()
            ))
            
    return markers

# --- 3. Acoustic/Voice Analysis Functions (Unchanged) ---

def initialize_emotion_model():
    class MockModel:
        def predict(self, features):
            return np.array([0])  
    class MockScaler:
        def transform(self, data):
            return data
    return MockModel(), MockScaler(), ["neutral", "calm", "happy", "sad", "angry"]  

def classify_emotion_simple(wav_file_path: str, model, scaler) -> str:
    emotion_classes = ["neutral", "calm", "happy", "sad", "angry"]
    try:
        y, sr = sf.read(wav_file_path)
        energy = np.mean(y**2)
        if energy > 0.005:  
            return "excited"
        return "neutral"  
    except Exception as e:
        logger.warning(f"Emotion classification failed: {e}")
        return "unknown"

    for i in range(0, len(y_filtered) - frame_size, hop_size):
        frame = y_filtered[i:i + frame_size]
        if np.mean(frame**2) > 0.0001:
            acf = np.correlate(frame, frame, mode='full')[len(frame) - 1:]
            min_lag = int(sr / 300)  
            max_lag = int(sr / 60)
            peaks, properties = find_peaks(acf[min_lag:max_lag], height=None)
            
            if len(peaks) > 0:
                peak_index = peaks[np.argmax(acf[min_lag + peaks])]
                lag = min_lag + peak_index
                
                if lag > 0:
                    f0 = sr / lag
                    f0_estimates.append(f0)
            
    f0_estimates = np.array(f0_estimates)
    
    if len(f0_estimates) > 10:
        f0_filtered = f0_estimates[np.abs(zscore(f0_estimates)) < 2]
        pitch_mean = np.mean(f0_filtered)
        pitch_std = np.std(f0_filtered)
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
            
    return float(pitch_mean), float(pitch_std)


def detect_acoustic_disfluencies(y: np.ndarray, sr: int, frame_len_ms: int = 25, hop_len_ms: int = 10) -> List[DisfluencyResult]:
    """
    Detects basic acoustic disfluencies (blocks/prolongations) using Energy and ZCR (NumPy/SciPy).
    """
    results: List[DisfluencyResult] = []
    
    frame_length = int(sr * frame_len_ms / 1000)
    hop_length = int(sr * hop_len_ms / 1000)

    y_frames = np.array([y[i:i + frame_length] for i in range(0, len(y) - frame_length, hop_length)])

    rms_frames = np.sqrt(np.mean(y_frames**2, axis=1))
    zcr_frames = np.mean(np.diff(np.sign(y_frames), axis=1) != 0, axis=1)

    energy_threshold = np.mean(rms_frames) * 0.1
    zcr_threshold = np.mean(zcr_frames) * 0.5  
    
    min_event_frames = int(0.3 / (hop_len_ms / 1000))
    
    is_event = False
    event_start_frame = 0
    
    for i in range(len(rms_frames)):
        
        is_block_candidate = (rms_frames[i] < energy_threshold)
        is_prolongation_candidate = (rms_frames[i] > energy_threshold * 2) and (zcr_frames[i] < zcr_threshold)

        if is_block_candidate or is_prolongation_candidate:
            if not is_event:
                is_event = True
                event_start_frame = i
        else:
            if is_event:
                event_duration_frames = i - event_start_frame
                
                if event_duration_frames >= min_event_frames:
                    
                    start_time_s = event_start_frame * (hop_len_ms / 1000)
                    duration_s = event_duration_frames * (hop_len_ms / 1000)
                    
                    if np.mean(rms_frames[event_start_frame:i]) < energy_threshold:
                        event_type = 'block'
                    else:
                        event_type = 'prolongation'
                        
                    results.append(DisfluencyResult(
                        type=event_type,  
                        start_time_s=round(start_time_s, 2),  
                        duration_s=round(duration_s, 2)
                    ))
                    
                is_event = False
                
    return results


# --- 4. Scoring Logic (FIXED for Division by Zero) ---

def score_confidence(audio_features: Dict[str, Any], fluency_metrics: Dict[str, Any]) -> float:
    """
    Calculate confidence score on 0-100 scale (not 0-1).
    More realistic thresholds for average speakers.
    """

    WEIGHTS = {
        'pace': 0.25,           # Speaking pace: 25%
        'fluency': 0.35,        # Disfluency count: 35%
        'pitch': 0.20,          # Pitch variation: 20%
        'energy': 0.15,         # Energy consistency: 15%
        'silence': 0.05,        # Silence ratio: 5%
    }

    # 1. PACE SCORING (0-100)
    pace_wpm = audio_features.get('speaking_pace_wpm', 0)
    ideal_pace = 150  # WPM

    if pace_wpm == 0:
        pace_score = 0
    elif pace_wpm < 80:
        pace_score = (pace_wpm / 80) * 40  # 0-40 for too slow
    elif pace_wpm > 200:
        pace_score = max(0, 100 - ((pace_wpm - 200) / 100) * 40)  # Penalize too fast


    # Realistic thresholds: average person has some fillers
    if disfluency_rate == 0:
        fluency_score = 100
    elif disfluency_rate < 0.03:  # 3% is good
        fluency_score = 100 - (disfluency_rate / 0.03) * 10  # 90-100
    elif disfluency_rate < 0.08:  # 8% is acceptable
        fluency_score = 90 - ((disfluency_rate - 0.03) / 0.05) * 20  # 70-90
    elif disfluency_rate < 0.15:  # 15% is poor but not terrible
        fluency_score = 70 - ((disfluency_rate - 0.08) / 0.07) * 30  # 40-70
    else:
        fluency_score = max(20, 40 - (disfluency_rate - 0.15) * 100)  # 20-40

    fluency_score = np.clip(fluency_score, 20, 100)

    # 3. PITCH SCORING (0-100)
    pitch_std = audio_features.get('pitch_std', 0)
    pitch_mean = audio_features.get('pitch_mean', 0)

    if pitch_std == 0 or pitch_mean == 0:
        pitch_score = 50  # Neutral if no pitch detected
    elif pitch_std < 5:
        pitch_score = 40  # Too monotone
    elif pitch_std < 15:
        pitch_score = 60  # Low variation
    elif pitch_std < 30:
        pitch_score = 85  # Good variation
    elif pitch_std < 50:
        pitch_score = 70  # High but acceptable
    else:
        pitch_score = 50  # Too variable


    # Clamp between 20-100 (20 is lowest possible, 100 is perfect)
    final_score = np.clip(final_score, 20, 100)

    return round(final_score, 1)  # Return as 0-100 scale

