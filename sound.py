import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import joblib

# Load the saved model and LabelEncoder
model = load_model('D:/MegaProject/saved_models/urban_sound_classifier.h5')
le = joblib.load('D:/MegaProject/saved_models/label_encoder.pkl')

# Function to extract features from audio data
def extract_features_from_audio(audio, sample_rate):
    try:
        # Convert multi-channel audio to mono by averaging channels
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, fmax=8000)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
        
        # Combine extracted features into a single array
        features = np.hstack((
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0),
            np.mean(contrast.T, axis=0),
            np.mean(tonnetz.T, axis=0)
        ))
        
        return features
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# Function to predict label of a sound
def predict_sound_label(audio, sample_rate):
    feature = extract_features_from_audio(audio, sample_rate)
    if feature is None:
        print(f"Unable to extract features from the audio")
        return None
    
    feature = feature.reshape(1, -1)
    predicted_vector = model.predict(feature)
    predicted_label = le.inverse_transform(np.argmax(predicted_vector, axis=1))
    return predicted_label[0]

# Function to record audio from microphone (using sounddevice)
def get_audio(duration=3, sample_rate=44100, device_index=5):
    global audio_data
    audio_data = []

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        global audio_data
        audio_data.append(indata.copy())

    # Adjust the channels parameter to 5 (0 through 4, inclusive)
    with sd.InputStream(samplerate=sample_rate, channels=5, device=device_index, callback=callback):
        print("Recording...")
        sd.sleep(duration * 1000)
        print("Recording complete")

    # Concatenate and convert to a numpy array
    audio = np.concatenate(audio_data)
    return audio, sample_rate

# Record audio from the microphone
audio, sample_rate = get_audio(duration=3, device_index=5)

# Predict the label for the recorded audio
predicted_label = predict_sound_label(audio, sample_rate)
print(f"The predicted label for the recorded sound is: {predicted_label}")
