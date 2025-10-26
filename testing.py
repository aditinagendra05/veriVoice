import os
import logging
from datetime import datetime
import csv
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from python_speech_features import mfcc

# configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defaults / thresholds
DURATION = 5
SAMPLE_RATE = 16000
MFCC_COUNT = 13
AUTH_THRESHOLD = 0.70
W_SIM = 0.7
W_PITCH = 0.3

# Example mapping: in real scenario, read from CSV/JSON
USER_MAP = {
    "aditi01": "Aditi",
    "bhavana02": "Bhavana"
}


def compute_pitch(signal: np.ndarray, sr: int) -> float:
    """Return estimated pitch (Hz) or 0.0 if not found."""
    if signal is None or len(signal) == 0:
        return 0.0
    signal = signal.astype(float)
    signal = signal - np.mean(signal)
    if np.allclose(signal, 0):
        return 0.0
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]
    d = np.diff(corr)
    candidates = np.where(d > 0)[0]
    if len(candidates) == 0:
        return 0.0
    start = candidates[0]
    peak = np.argmax(corr[start:]) + start
    if peak == 0:
        return 0.0
    pitch = float(sr) / float(peak)
    if pitch < 50 or pitch > 500:
        return 0.0
    return float(pitch)


def cosine_similarity(a, b) -> float:
    a = np.array(a).astype(float)
    b = np.array(b).astype(float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def record_test_audio(user_id: str, duration: int = DURATION, sr: int = SAMPLE_RATE) -> str:
    """Record short test audio for a user and return saved filepath."""
    template_dir = os.path.join("data", user_id, "template")
    os.makedirs(template_dir, exist_ok=True)
    filename = os.path.join(template_dir, "test.wav")
    logger.info("Recording test audio for %s (%ds)...", user_id, duration)
    try:
        audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        # audio_data shape is (n_frames, 1); soundfile accepts that
        sf.write(filename, audio_data, sr)
        logger.info("Test audio saved at: %s", filename)
        return filename
    except Exception as e:
        logger.error("Failed to record audio: %s", e)
        raise


def extract_features(filename: str) -> np.ndarray:
    """Extract MFCC mean + pitch from a file and return feature vector (length MFCC_COUNT+1)."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Audio file not found: {filename}")
    y, sr = sf.read(filename)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if len(y) == 0:
        raise ValueError("Audio file contains no samples")
    try:
        mfcc_feat = mfcc(y, samplerate=sr, numcep=MFCC_COUNT, winlen=0.025, winstep=0.01)
        mfcc_mean = np.mean(mfcc_feat, axis=0)
    except Exception as e:
        logger.error("MFCC extraction failed: %s", e)
        raise
    pitch_value = compute_pitch(y, sr)
    return np.append(mfcc_mean, pitch_value)


def log_attendance(user_id: str, auth_decision: str, spoof_score: float) -> None:
    """Append attendance rows to user and admin CSV logs."""
    username = USER_MAP.get(user_id, "Unknown")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # User log
    user_log_dir = os.path.join("data", user_id, "logs")
    os.makedirs(user_log_dir, exist_ok=True)
    user_log_file = os.path.join(user_log_dir, "attendance.csv")
    try:
        with open(user_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_id, username, auth_decision, f"{spoof_score:.2f}"])
    except Exception as e:
        logger.error("Failed to write user log: %s", e)

    # Admin log
    try:
        os.makedirs("admin_logs", exist_ok=True)
        admin_log_file = os.path.join("admin_logs", "overall_attendance.csv")
        with open(admin_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_id, username, auth_decision, f"{spoof_score:.2f}"])
    except Exception as e:
        logger.error("Failed to write admin log: %s", e)

    logger.info("Attendance logged for %s at %s", username, timestamp)


def authenticate_user(user_id: str, test_audio_path: Optional[str] = None) -> dict:
    """
    High-level function that runs the authentication flow.
    Returns a result dict with similarity, decision, spoof_score, paths.
    """
    features_path = os.path.join("data", user_id, "features", "features.npy")
    if not os.path.exists(features_path):
        raise FileNotFoundError("Template features not found. Run extraction.py first.")

    # Record if no test_audio_path provided
    if test_audio_path is None:
        test_audio_path = record_test_audio(user_id)

    test_features = extract_features(test_audio_path)
    template_features = np.load(features_path)

    # Validate shapes
    if template_features.shape != test_features.shape:
        logger.warning("Feature vector size mismatch (template %s vs test %s). Attempting to align by truncation/padding.",
                       template_features.shape, test_features.shape)
        # simple alignment: truncate or pad with zeros
        max_len = max(template_features.size, test_features.size)
        tf = np.zeros(max_len, dtype=float)
        xf = np.zeros(max_len, dtype=float)
        tf[:template_features.size] = template_features
        xf[:test_features.size] = test_features
        template_features = tf
        test_features = xf

    sim = cosine_similarity(template_features, test_features)
    auth_decision = "ALLOW" if sim >= AUTH_THRESHOLD else "DENY"
    pitch_diff = abs(float(template_features[-1]) - float(test_features[-1]))
    spoof_score = W_SIM * (1 - sim) + W_PITCH * min(1.0, pitch_diff/200.0)
    spoof_score = float(np.clip(spoof_score * 10, 0, 10))

    # Log attendance
    try:
        log_attendance(user_id, auth_decision, spoof_score)
    except Exception:
        logger.exception("Logging failed")

    result = {
        "user_id": user_id,
        "similarity": float(sim),
        "decision": auth_decision,
        "spoof_score": float(spoof_score),
        "test_audio": test_audio_path,
        "template_features": features_path
    }
    return result


def main():
    try:
        user_id = input("Enter your user ID (e.g., aditi01): ").strip()
        if user_id == "":
            print("User ID required.")
            return
        res = authenticate_user(user_id)
        print("\n=== Attendance Authentication ===")
        print(f"Similarity: {res['similarity']:.3f}, Decision: {res['decision']}, Spoof score: {res['spoof_score']:.2f}")
    except Exception as e:
        logger.exception("Authentication failed: %s", e)


if __name__ == "__main__":
    main()
