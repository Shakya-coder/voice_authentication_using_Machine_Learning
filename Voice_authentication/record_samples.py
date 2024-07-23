#Code by R K Shakya

import os
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

def record_audio(filename, duration=5, sample_rate=44100):
    print(f"Recording audio for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    print("Finished recording.")
    wav.write(filename, sample_rate, audio)

if __name__ == "__main__":
    folder_path = './training_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    num_samples = 25  # Number of samples to record 
    for i in range(num_samples):
        input(f"Press Enter to start recording sample {i + 1}...")
        filename = os.path.join(folder_path, f'reference_{i + 1}.wav')
        record_audio(filename, duration=5)
        print(f"Sample {i + 1} recorded.\n")
