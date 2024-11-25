import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib

# Load the saved model
model = load_model('D:/Sound_Recognition/New_Model/df/Sound_Classifier_CNN.h5')

# Load the saved LabelEncoder
le = joblib.load('D:/Sound_Recognition/New_Model/df/Sound_label.pkl')

# Function to extract features from audio files
def extract_features(file_name, target_size=195):
    try:
        audio, sample_rate = librosa.load(file_name, sr=44100)
        
        # Extract different features from the audio file
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

        # Pad or truncate the features array to match the required input size of the model
        if len(features) < target_size:
            # Pad with zeros if the feature size is less than target size (195)
            features = np.pad(features, (0, target_size - len(features)), 'constant')
        elif len(features) > target_size:
            # Truncate if the feature size is larger than target size
            features = features[:target_size]
        
        return features
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

# Function to predict the label of a sound file with confidence threshold
def predict_sound_label(file_name, confidence_threshold=0.5):
    feature = extract_features(file_name)
    if feature is None:
        print(f"Unable to extract features from {file_name}")
        return None
    
    # Reshape the feature for CNN input: (batch_size, height, width, channels)
    feature = feature.reshape(1, feature.shape[0], 1, 1)  # Reshaping to match model input
    
    # Predict the label using the CNN model
    predicted_vector = model.predict(feature)
    
    # Get the confidence score (maximum probability)
    confidence_score = np.max(predicted_vector)
    
    # Check if the confidence score is above the threshold
    if confidence_score >= confidence_threshold:
        # Get the label with the highest confidence score
        predicted_label = le.inverse_transform(np.argmax(predicted_vector, axis=1))
        return predicted_label[0]
    else:
        # If confidence is below the threshold, return "Uncertain"
        return "Uncertain"

# Test with a new or random sound file
random_file_name = 'C:/Users/yoges/Downloads/10.mp3'
predicted_label = predict_sound_label(random_file_name)

print(f"The predicted label for the sound is: {predicted_label}")
