import numpy as np
import librosa
import joblib
import sounddevice as sd
import scipy.io.wavfile as wav

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def load_model(model_filename):
    try:
        model = joblib.load(model_filename)
        return model
    except Exception as e:
        print(f"Error loading model from {model_filename}: {e}")
        return None

def record_audio(filename, duration=5, sample_rate=44100):
    print(f"Recording audio for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    print("Finished recording.")
    wav.write(filename, sample_rate, audio)

def authenticate_voice_sample(model, sample_filename):
    try:
        features = extract_features(sample_filename)
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        print(f"Error predicting voice sample: {e}")
        return None

if __name__ == "__main__":
    # Load the trained model
    model_filename = 'voice_authentication_model.pkl'
    model = load_model(model_filename)
    if model is None:
        exit(1)

    # Record a voice sample for authentication
    sample_filename = 'authenticate_sample.wav'
    record_audio(sample_filename, duration=5)  # Adjust duration as needed

    # Authenticate the recorded voice sample
    prediction = authenticate_voice_sample(model, sample_filename)

    # Print authentication result
    if prediction == 1:
        print("Authenticated: Voice matches.")
    else:
        print("Not Authenticated: Voice does not match.")
