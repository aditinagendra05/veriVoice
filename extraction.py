import os
import logging
import numpy as np
import soundfile as sf
from python_speech_features import mfcc
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default settings
MFCC_COUNT = 13

def compute_pitch(signal: np.ndarray, sr: int) -> float:
    """
    Simple pitch estimator using autocorrelation.
    Returns 0.0 if no reliable pitch is found.
    """
    if signal is None or len(signal) == 0:
        return 0.0

    signal = signal.astype(float) - float(np.mean(signal))
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

def _load_mono(path: str):
    """
    Load audio and return (y, sr) with mono signal (1D numpy array).
    Raises FileNotFoundError if path missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, sr = sf.read(path)
    # Convert to mono if needed
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    # Ensure float dtype
    y = y.astype(np.float32)
    return y, sr

def extract_features(user_id: Optional[str] = None, audio_path: Optional[str] = None) -> str:
    """
    Loads the template audio for user_id (or uses audio_path if provided),
    extracts MFCC mean + pitch, saves as features.npy under data/<user_id>/features/
    Returns the saved features path.

    Usage:
      extract_features("aditi01")
      or
      extract_features(audio_path="uploads/recording_2025.wav", user_id="aditi01")
    """
    if audio_path is None:
        if user_id is None:
            raise ValueError("Either user_id or audio_path must be provided")
        template_path = os.path.join("data", user_id, "template", "template.wav")
    else:
        template_path = audio_path
        # if user_id not provided, infer features dir based on audio location
        if user_id is None:
            # fallback to saving features next to audio (folder features/)
            base_dir = os.path.dirname(template_path) or "."
            features_dir = os.path.join(base_dir, "features")
            os.makedirs(features_dir, exist_ok=True)
            features_path = os.path.join(features_dir, "features.npy")
        else:
            features_dir = os.path.join("data", user_id, "features")
            os.makedirs(features_dir, exist_ok=True)
            features_path = os.path.join(features_dir, "features.npy")

    if audio_path is None:
        features_dir = os.path.join("data", user_id, "features")
        os.makedirs(features_dir, exist_ok=True)
        features_path = os.path.join(features_dir, "features.npy")

    # Load audio
    try:
        y, sr = _load_mono(template_path)
    except Exception as e:
        logger.error("Failed to load audio: %s", e)
        raise

    if len(y) < 10:
        logger.warning("Audio is very short (%d samples). MFCC may be unreliable.", len(y))

    # Extract MFCC mean
    try:
        mfcc_feat = mfcc(y, samplerate=sr, numcep=MFCC_COUNT, winlen=0.025, winstep=0.01)
        mfcc_mean = np.mean(mfcc_feat, axis=0)
    except Exception as e:
        logger.error("MFCC extraction failed: %s", e)
        raise

    # Extract pitch
    pitch_value = compute_pitch(y, sr)

    # Combine MFCC + pitch
    feature_vector = np.append(mfcc_mean, pitch_value).astype(np.float32)

    # Save features
    np.save(features_path, feature_vector)
    logger.info("Features saved at: %s", features_path)

    return features_path

# Run directly for testing
if __name__ == "__main__":
    uid = input("Enter your user ID (e.g., aditi01) or leave empty to enter a file path: ").strip()
    if uid == "":
        path = input("Enter audio file path: ").strip()
        out = extract_features(audio_path=path)
    else:
        out = extract_features(user_id=uid)
    print(f"Saved features to: {out}")
