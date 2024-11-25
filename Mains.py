import pandas as pd
import numpy as np
import librosa

# Load metadata
metadata = pd.read_csv("D:/MegaProjecturbansound8k-metadata.json")

# Function to extract features from audio files
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load("D:/MegaProject/archive", sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error processing {"D:/MegaProject/archive"}: {e}")
        return None

# Example of processing files
features = []
labels = []

for index, row in metadata.iterrows():
    file_name = os.path.join('path_to_audio_files', row["slice_file_name"])
    label = row["class"]
    feature = extract_features(file_name)
    if feature is not None:
        features.append(feature)
        labels.append(label)

X = np.array(features)
y = np.array(labels)
