import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np

def record_voice(filename, duration=5, samplerate=44100):
    """
    Records audio from the microphone and saves it as a WAV file.
    """
    print(f"🎙 Recording {filename} for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Saved: {filename}\n")

def extract_features(file):
    """
    Extracts combined MFCC, Delta, and Delta-Delta features from an audio file.
    """
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    mean_features = np.mean(combined, axis=1)
    return mean_features

def compare_voices(file1, file2, threshold=50.0):
    """
    Compares two voice recordings and prints similarity score.
    Lower score → more similar.
    """
    f1 = extract_features(file1)
    f2 = extract_features(file2)

    similarity = np.linalg.norm(f1 - f2)  # Euclidean distance

    print(f"🎯 Voice similarity score: {similarity:.2f}")
    if similarity < threshold:
        print("✅ The voices are likely similar.")
    else:
        print("❌ The voices are likely different.")
    return similarity

if __name__ == "__main__":
    # Record two voices
    record_voice("voice1.wav", duration=5)
    record_voice("voice2.wav", duration=5)

    # Compare them
    compare_voices("voice1.wav", "voice2.wav")

