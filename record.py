import os
import sounddevice as sd
import soundfile as sf

# Default settings
DURATION = 10         # seconds
SAMPLE_RATE = 16000   # 16 kHz recommended

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
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording finished!")

    # Save audio
    sf.write(filename, audio_data, sr)
    print(f"💾 Audio saved at: {filename}")

    return filename

# Run directly (for testing without UI)
if __name__ == "__main__":
    user_id = input("Enter your user ID (e.g., aditi01): ").strip()
    record_user_audio(user_id)
