import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Step 1: Load the Dataset
df = pd.read_csv('D:/Sound_Recognition/ESC-50-master/ESC-50-master/meta/esc50.csv')

# Debugging: Check DataFrame structure
print(df.columns)
print(df.head())

# Define the base path for audio files
audio_base_path = 'D:/Sound_Recognition/ESC-50-master/ESC-50-master/audio/'

# Step 2: Preprocess Audio Data
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None, duration=3)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Apply feature extraction to each audio file
# Create full file paths for audio files
full_file_paths = [os.path.join(audio_base_path, row['filename']) for _, row in df.iterrows()]

# Extract features
X = np.array([extract_features(file) for file in full_file_paths])
y = df['target'].values  # Use the target column for labels

# Remove any None values from X and corresponding labels in y
X = np.array([x for x in X if x is not None])
y = y[:len(X)]

# Step 3: Prepare Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Split the Dataset
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Debugging: Check shapes
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Step 5: Build the Model
model = Sequential()
model.add(Flatten(input_shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Step 7: Save the Model
model.save('ESC_sound_recognition_model.h5')
print("Model saved as ESC_sound_recognition_model.h5")

# Step 8: Save Label Mappings
# Create a mapping of encoded labels to original labels
label_mapping = pd.DataFrame({
    'encoded_label': range(len(label_encoder.classes_)),
    'original_label': label_encoder.classes_
})

# Save the label mapping to a CSV file
label_mapping.to_csv('label_mapping.csv', index=False)
print("Label mappings saved as label_mapping.csv")

# Optional: Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
