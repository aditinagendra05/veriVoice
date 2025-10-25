import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from python_speech_features import mfcc
from datetime import datetime
import csv

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

def compute_pitch(signal, sr):
    signal = signal - np.mean(signal)
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
    return pitch

def cosine_similarity(a, b):
    a = np.array(a).astype(float)
    b = np.array(b).astype(float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom

def record_test_audio(user_id, duration=DURATION, sr=SAMPLE_RATE):
    filename = os.path.join("data", user_id, "template", "test.wav")
    print(f"\n🎙️ Recording test audio for {user_id}...")
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio_data, sr)
    print(f"✅ Test audio saved at: {filename}")
    return filename

def extract_features(filename):
    y, sr = sf.read(filename)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    mfcc_feat = mfcc(y, samplerate=sr, numcep=MFCC_COUNT, winlen=0.025, winstep=0.01)
    mfcc_mean = np.mean(mfcc_feat, axis=0)
    pitch_value = compute_pitch(y, sr)
    return np.append(mfcc_mean, pitch_value)

def log_attendance(user_id, auth_decision, spoof_score):
    username = USER_MAP.get(user_id, "Unknown")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # User log
    user_log_dir = os.path.join("data", user_id, "logs")
    os.makedirs(user_log_dir, exist_ok=True)
    user_log_file = os.path.join(user_log_dir, "attendance.csv")
    with open(user_log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, user_id, username, auth_decision, f"{spoof_score:.2f}"])

    # Admin log
    os.makedirs("admin_logs", exist_ok=True)
    admin_log_file = os.path.join("admin_logs", "overall_attendance.csv")
    with open(admin_log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, user_id, username, auth_decision, f"{spoof_score:.2f}"])

    print(f"📝 Attendance logged for {username} at {timestamp}")

def main():
    user_id = input("Enter your user ID (e.g., aditi01): ").strip()

    # Check template
    features_path = os.path.join("data", user_id, "features", "features.npy")
    if not os.path.exists(features_path):
        print("❌ Template features not found! Run extraction.py first.")
        return

    # Record and extract test audio
    test_file = record_test_audio(user_id)
    test_features = extract_features(test_file)
    template_features = np.load(features_path)

    # Compute similarity
    sim = cosine_similarity(template_features, test_features)
    auth_decision = "ALLOW" if sim >= AUTH_THRESHOLD else "DENY"
    pitch_diff = abs(template_features[-1] - test_features[-1])
    spoof_score = W_SIM * (1 - sim) + W_PITCH * min(1.0, pitch_diff/200.0)
    spoof_score = np.clip(spoof_score * 10, 0, 10)

    # Print results
    print("\n=== Attendance Authentication ===")
    print(f"Similarity: {sim:.3f}, Decision: {auth_decision}, Spoof score: {spoof_score:.2f}")

    # Log attendance
    log_attendance(user_id, auth_decision, spoof_score)

if __name__ == "__main__":
    main()
