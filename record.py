import os
import time
import queue
import threading

import sounddevice as sd
import soundfile as sf

# Default settings
DURATION = 10         # seconds
SAMPLE_RATE = 16000   # 16 kHz recommended


ROOT_DIR = os.path.dirname(__file__)
_UPLOADS_DIR = os.path.join(ROOT_DIR, "uploads")
os.makedirs(_UPLOADS_DIR, exist_ok=True)

_state = {
    "stream": None,
    "queue": None,
    "thread": None,
    "file": None,
    "output_path": None,
    "recording": False,
}


def _writer_thread_func(sf_file: sf.SoundFile, q: queue.Queue):
    try:
        while _state["recording"] or not q.empty():
            try:
                frames = q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                sf_file.write(frames)
            except Exception:
                # silently continue on writer error to avoid crashing the callback
                pass
    finally:
        try:
            sf_file.close()
        except Exception:
            pass


def start_recording(output_path: str = None, samplerate: int = SAMPLE_RATE, channels: int = 1, subtype: str = "PCM_16") -> str:
    """
    Start background recording and return the path that will be written.
    Raises RuntimeError if already recording.
    """
    if _state["recording"]:
        raise RuntimeError("Already recording")

    if output_path is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(_UPLOADS_DIR, f"recording_{ts}.wav")
    else:
        os.makedirs(os.path.dirname(output_path) or _UPLOADS_DIR, exist_ok=True)

    q = queue.Queue()

    sf_file = sf.SoundFile(output_path, mode="w", samplerate=samplerate, channels=channels, subtype=subtype)

    def callback(indata, frames, time_info, status):
        # indata is numpy array (frames, channels)
        try:
            q.put(indata.copy(), block=False)
        except queue.Full:
            # drop frames if queue full to keep callback non-blocking
            pass

    stream = sd.InputStream(samplerate=samplerate, channels=channels, dtype="float32", callback=callback, blocksize=1024)

    writer = threading.Thread(target=_writer_thread_func, args=(sf_file, q), daemon=True)

    _state.update({
        "stream": stream,
        "queue": q,
        "thread": writer,
        "file": sf_file,
        "output_path": output_path,
        "recording": True,
    })

    stream.start()
    writer.start()
    return output_path


def stop_recording(timeout: float = 5.0) -> str:
    """
    Stop background recording, flush writer and return saved file path.
    Raises RuntimeError if not recording.
    """
    if not _state["recording"]:
        raise RuntimeError("Not recording")

    _state["recording"] = False
    try:
        _state["stream"].stop()
    except Exception:
        pass
    try:
        _state["stream"].close()
    except Exception:
        pass

    thr = _state.get("thread")
    if thr:
        thr.join(timeout=timeout)

    path = _state.get("output_path")

    # ensure file closed
    try:
        f = _state.get("file")
        if f is not None and not f.closed:
            f.close()
    except Exception:
        pass

    # reset state
    _state.update({
        "stream": None,
        "queue": None,
        "thread": None,
        "file": None,
        "output_path": None,
        "recording": False,
    })
    return path


def is_recording() -> bool:
    return bool(_state.get("recording"))


def record_user_audio(user_id, duration=DURATION, sr=SAMPLE_RATE):
    """
    Records voice for a given user_id and saves it inside:
    data/<user_id>/template/template.wav
    """
    # Build folder path and create if it doesn't exist
    template_dir = os.path.join("data", user_id, "template")
    os.makedirs(template_dir, exist_ok=True)

    # File path
    filename = os.path.join(template_dir, "template.wav")

    print(f"\n🎙️ Recording for {user_id}... Speak clearly for {duration} seconds.")

    # Use background API and block for duration to preserve original behavior
    start_recording(output_path=filename, samplerate=sr, channels=1)
    try:
        time.sleep(duration)
    finally:
        saved = stop_recording()

    print("✅ Recording finished!")
    print(f"💾 Audio saved at: {saved}")

    return saved
# Run directly (for testing without UI)
if __name__ == "__main__":
    user_id = input("Enter your user ID (e.g., aditi01): ").strip()
    record_user_audio(user_id)
