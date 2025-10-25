import os
import numpy as np
import soundfile as sf
from python_speech_features import mfcc

# Default settings
MFCC_COUNT = 13

def compute_pitch(signal, sr):
    """
    Simple pitch estimator using autocorrelation.
    """
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

def extract_features(user_id):
    """
    Loads the template audio for user_id, extracts MFCC mean + pitch,
    and saves as features.npy under data/<user_id>/features/
    """
    template_path = os.path.join("data", user_id, "template", "template.wav")
    features_dir = os.path.join("data", user_id, "features")
    os.makedirs(features_dir, exist_ok=True)
    features_path = os.path.join(features_dir, "features.npy")

    # Load audio
    y, sr = sf.read(template_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Extract MFCC mean
    mfcc_feat = mfcc(y, samplerate=sr, numcep=MFCC_COUNT, winlen=0.025, winstep=0.01)
    mfcc_mean = np.mean(mfcc_feat, axis=0)

    # Extract pitch
    pitch_value = compute_pitch(y, sr)

    # Combine MFCC + pitch
    feature_vector = np.append(mfcc_mean, pitch_value)

    # Save features
    np.save(features_path, feature_vector)
    print(f"💾 Features saved at: {features_path}")

    return features_path

# Run directly for testing
if __name__ == "__main__":
    user_id = input("Enter your user ID (e.g., aditi01): ").strip()
    extract_features(user_id)
