import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
import joblib

# Load metadata
metadata = pd.read_csv('D:/Sound_Recognition/archive/Hear4U_dataset/meta/hear4u.csv')

# Function to extract features from audio files
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=44100)
        n_fft = min(1024, len(audio))  # Ensure n_fft does not exceed the length of the audio
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, fmax=8000, n_fft=n_fft)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=n_fft)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=n_fft)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
        
        features = np.hstack((  # Stack all features together
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0),
            np.mean(contrast.T, axis=0),
            np.mean(tonnetz.T, axis=0)
        ))
        
        return features
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

# Extract features and labels from the dataset
new_features = []
new_labels = []

for index, row in metadata.iterrows():
    file_name = os.path.join('D:/Sound_Recognition/archive/Hear4U_dataset', row["filename"])
    if not os.path.exists(file_name):
        print(f"File does not exist: {file_name}")
        continue

    print(f"Processing file: {file_name}")
    feature = extract_features(file_name)

    if feature is not None:
        new_features.append(feature)
        new_labels.append(row["category"])  # Adjust column name to match your CSV
    else:
        print(f"Failed to extract features from {file_name}")

# Convert lists to numpy arrays
X = np.array(new_features)
y = np.array(new_labels)

# Encode labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)

# Build a more complex model
model = Sequential()
model.add(Dense(512, input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(len(le.classes_)))  # Number of output classes
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Save the trained model and LabelEncoder
model.save('D:/Sound_Recognition/Model_Sound/Sound_Classifier.h5')
joblib.dump(le, 'D:/Sound_Recognition/Model_Sound/Sound_label.pkl')

print("Model and LabelEncoder have been saved successfully.")

# Function to predict label of a sound file
def predict_sound_label(file_name):
    feature = extract_features(file_name)
    if feature is None:
        print(f"Unable to extract features from {file_name}")
        return None
    
    feature = feature.reshape(1, -1)
    predicted_vector = model.predict(feature)
    predicted_label = le.inverse_transform(np.argmax(predicted_vector, axis=1))
    return predicted_label[0]

# Test with a new or random sound file
random_file_name = 'D:/Sound_Recognition/archive/fold1/7061-6-0-0.wav'
predicted_label = predict_sound_label(random_file_name)
print(f"The predicted label for the sound is: {predicted_label}")
