from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import soundfile as sf
from python_speech_features import mfcc
from datetime import datetime
import csv
import sys
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("veriVoice-backend")

# ensure project root is importable so we can import modules at repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import extraction
    import record
    import testing
except Exception as e:
    logger.exception("Failed to import project modules: %s", e)
    raise

app = FastAPI(title="veriVoice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DURATION = 5
SAMPLE_RATE = 16000
MFCC_COUNT = 13
AUTH_THRESHOLD = 0.70
W_SIM = 0.7
W_PITCH = 0.3

# User mapping - you can load this from a database or CSV file
USER_MAP = {
    "aditi01": "Aditi",
    "bhavana02": "Bhavana"
}

def compute_pitch(signal, sr):
    """Simple pitch estimator using autocorrelation."""
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
    """Calculate cosine similarity between two vectors."""
    a = np.array(a).astype(float)
    b = np.array(b).astype(float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom

def extract_features(filename):
    """Extract MFCC and pitch features from audio file."""
    y, sr = sf.read(filename)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    
    mfcc_feat = mfcc(y, samplerate=sr, numcep=MFCC_COUNT, 
                     winlen=0.025, winstep=0.01)
    mfcc_mean = np.mean(mfcc_feat, axis=0)
    pitch_value = compute_pitch(y, sr)
    
    return np.append(mfcc_mean, pitch_value)

def log_attendance(user_id, auth_decision, spoof_score, similarity):
    """Log attendance to both user and admin logs."""
    username = USER_MAP.get(user_id, "Unknown")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # User log
    user_log_dir = os.path.join("data", user_id, "logs")
    os.makedirs(user_log_dir, exist_ok=True)
    user_log_file = os.path.join(user_log_dir, "attendance.csv")
    
    with open(user_log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, user_id, username, auth_decision, 
                        f"{spoof_score:.2f}", f"{similarity:.3f}"])

    # Admin log
    os.makedirs("admin_logs", exist_ok=True)
    admin_log_file = os.path.join("admin_logs", "overall_attendance.csv")
    
    with open(admin_log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, user_id, username, auth_decision, 
                        f"{spoof_score:.2f}", f"{similarity:.3f}"])

    print(f"📝 Attendance logged for {username} at {timestamp}")

@app.route('/verify', methods=['POST'])
def verify_voice():
    """Main endpoint for voice verification."""
    try:
        # Get user_id and audio file
        user_id = request.form.get('user_id')
        audio_file = request.files.get('audio')
        
        if not user_id or not audio_file:
            return jsonify({
                'success': False,
                'message': 'Missing user_id or audio file'
            }), 400
        
        # Check if template features exist
        features_path = os.path.join("data", user_id, "features", "features.npy")
        if not os.path.exists(features_path):
            return jsonify({
                'success': False,
                'message': 'User template not found. Please register first.'
            }), 404
        
        # Save uploaded audio temporarily
        temp_dir = os.path.join("data", user_id, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        test_file = os.path.join(temp_dir, "test.wav")
        audio_file.save(test_file)
        
        # Extract features from test audio
        test_features = extract_features(test_file)
        template_features = np.load(features_path)
        
        # Compute similarity
        similarity = cosine_similarity(template_features, test_features)
        auth_decision = "ALLOW" if similarity >= AUTH_THRESHOLD else "DENY"
        
        # Calculate spoof score
        pitch_diff = abs(template_features[-1] - test_features[-1])
        spoof_score = W_SIM * (1 - similarity) + W_PITCH * min(1.0, pitch_diff/200.0)
        spoof_score = np.clip(spoof_score * 10, 0, 10)
        
        # Log attendance
        log_attendance(user_id, auth_decision, spoof_score, similarity)
        
        # Clean up temp file
        os.remove(test_file)
        
        # Return response
        return jsonify({
            'success': auth_decision == "ALLOW",
            'decision': auth_decision,
            'similarity': float(similarity),
            'spoof_score': float(spoof_score),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message': 'Voice verified successfully' if auth_decision == "ALLOW" 
                      else 'Voice verification failed'
        })
        
    except Exception as e:
        print(f"Error in verify_voice: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/register', methods=['POST'])
def register_user():
    """Endpoint to register a new user with their voice template."""
    try:
        user_id = request.form.get('user_id')
        audio_file = request.files.get('audio')
        
        if not user_id or not audio_file:
            return jsonify({
                'success': False,
                'message': 'Missing user_id or audio file'
            }), 400
        
        # Save template audio
        template_dir = os.path.join("data", user_id, "template")
        os.makedirs(template_dir, exist_ok=True)
        template_path = os.path.join(template_dir, "template.wav")
        audio_file.save(template_path)
        
        # Extract and save features
        y, sr = sf.read(template_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        
        mfcc_feat = mfcc(y, samplerate=sr, numcep=MFCC_COUNT, 
                        winlen=0.025, winstep=0.01)
        mfcc_mean = np.mean(mfcc_feat, axis=0)
        pitch_value = compute_pitch(y, sr)
        feature_vector = np.append(mfcc_mean, pitch_value)
        
        features_dir = os.path.join("data", user_id, "features")
        os.makedirs(features_dir, exist_ok=True)
        features_path = os.path.join(features_dir, "features.npy")
        np.save(features_path, feature_vector)
        
        return jsonify({
            'success': True,
            'message': f'User {user_id} registered successfully',
            'features_path': features_path
        })
        
    except Exception as e:
        print(f"Error in register_user: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Registration failed: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

async def _save_upload(upload: UploadFile, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    filename = upload.filename or f"upload_{int(__import__('time').time())}.wav"
    # avoid overwrite: append timestamp if exists
    out_path = os.path.join(dest_dir, filename)
    if os.path.exists(out_path):
        name, ext = os.path.splitext(filename)
        out_path = os.path.join(dest_dir, f"{name}_{int(__import__('time').time())}{ext}")
    data = await upload.read()
    with open(out_path, "wb") as wf:
        wf.write(data)
    return out_path


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/record/start")
async def api_start_recording(
    output_name: Optional[str] = Form(None),
    samplerate: Optional[int] = Form(None),
    channels: Optional[int] = Form(None),
):
    """Start server-side recording. Returns path being recorded to."""
    try:
        sr = int(samplerate) if samplerate is not None else getattr(record, "SAMPLE_RATE", 16000)
        ch = int(channels) if channels is not None else 1
        out = None
        if output_name:
            uploads_dir = os.path.join(PROJECT_ROOT, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            out = os.path.join(uploads_dir, output_name)
        path = record.start_recording(output_path=out, samplerate=sr, channels=ch)
        return JSONResponse({"status": "recording_started", "path": path})
    except Exception as e:
        logger.exception("start_recording failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/record/stop")
async def api_stop_recording():
    """Stop server-side recording. Returns saved file path."""
    try:
        path = record.stop_recording()
        return JSONResponse({"status": "recording_stopped", "path": path})
    except Exception as e:
        logger.exception("stop_recording failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
async def api_extract(
    file: Optional[UploadFile] = File(None),
    user_id: Optional[str] = Form(None),
):
    """
    Extract features.
    - If file is uploaded: save to uploads/ and extract_features(audio_path=...)
    - Else if user_id provided: extract stored template for that user
    """
    try:
        if file is not None:
            saved = await _save_upload(file, os.path.join(PROJECT_ROOT, "uploads"))
            features_path = extraction.extract_features(user_id=user_id, audio_path=saved)
            return {"status": "extracted", "features_path": features_path}
        else:
            if not user_id:
                raise HTTPException(status_code=400, detail="Provide either an uploaded file or a user_id")
            features_path = extraction.extract_features(user_id=user_id)
            return {"status": "extracted", "features_path": features_path}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test")
async def api_test(
    user_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Authenticate / test:
    - Provide user_id (required).
    - Optionally upload a test audio file; if not provided, server will record a short test audio using testing.record_test_audio().
    """
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required for testing/authentication")
        test_path = None
        if file is not None:
            test_path = await _save_upload(file, os.path.join(PROJECT_ROOT, "uploads"))
        result = testing.authenticate_user(user_id, test_audio_path=test_path)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Authentication/test failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("admin_logs", exist_ok=True)
    
    print("🚀 Starting veriVoice Backend Server...")
    print("📡 Server running on http://localhost:5000")
    print("🎤 Ready to accept voice verification requests")
    
    app.run(debug=True, host='0.0.0.0', port=5000)