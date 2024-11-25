import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib

# Load the model and label encoder
model = load_model('D:/Sound_Recognition/Model_Sound/NEW_MODEL/NEW_MODEL_SOUND.h5')
le = joblib.load('D:/Sound_Recognition/Model_Sound/NEW_MODEL/NEW_MODEL_SOUND.pkl')

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=44100)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, fmax=8000)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)

        # Ensure the feature vector has 193 elements
        features = np.hstack((
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0)[:128],
            np.mean(contrast.T, axis=0),
            np.mean(tonnetz.T, axis=0)
        ))

        return features
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

def predict_sound_label(file_name):
    feature = extract_features(file_name)
    if feature is None:
        print(f"Unable to extract features from {file_name}")
        return None

    feature = feature.reshape(1, -1)
    predicted_vector = model.predict(feature)
    predicted_label = le.inverse_transform([np.argmax(predicted_vector, axis=1)[0]])
    return predicted_label[0]

# Test with a sound file
random_file_name = 'C:/Users/yoges/Downloads/99.mp3'
predicted_label = predict_sound_label(random_file_name)
print(f"The predicted label for the sound is: {predicted_label}")
