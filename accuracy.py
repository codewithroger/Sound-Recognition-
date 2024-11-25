import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset from the CSV file
csv_file_path = 'D:/Sound_Recognition/archive/UrbanSound8K.csv'
data = pd.read_csv(csv_file_path)

# Assuming the audio features are in 'feature_column_name' and labels in 'label_column_name'
# Adjust these names based on your actual CSV structure
feature_column_name = 'feature_column_name'  # Replace with your actual feature column name
label_column_name = 'label_column_name'      # Replace with your actual label column name

# Prepare features and labels
X = np.array(data[feature_column_name].tolist())  # Replace this with your actual feature extraction logic
y = data[label_column_name].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load Pre-trained Model
model = load_model('D:/Sound_Recognition/New_Model/Sound_Classifier_CNN.h5')

# Fine-tune the Model (Unfreeze layers for fine-tuning)
for layer in model.layers[:5]:
    layer.trainable = False  # Freeze the first few layers
for layer in model.layers[5:]:
    layer.trainable = True   # Unfreeze the rest of the layers

# Data Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    rotation_range=10,
    horizontal_flip=True  # Optional, if your data supports flipping
)

# Apply data augmentation during training
train_generator = datagen.flow(X_train.reshape(X_train.shape[0], 40, 1, 1), y_train, batch_size=32)  # Adjust the shape as needed

# Add Regularization (L2 Regularization and Dropout)
for layer in model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer = regularizers.l2(0.001)  # L2 regularization

# Add dropout if needed to prevent overfitting
for layer in model.layers:
    if isinstance(layer, Dropout):
        layer.rate = 0.3  # Adjust dropout rate

# Hyperparameter Tuning using Keras Tuner
def model_builder(hp):
    for layer in model.layers[:5]:
        layer.trainable = False  # Keep some layers frozen

    for layer in model.layers[5:]:
        layer.trainable = True  # Unfreeze layers for tuning

    # Add additional Dense layers for more fine-tuning
    x = model.output
    x = Dense(hp.Int('units', min_value=128, max_value=512, step=128), activation='relu')(x)
    x = Dropout(hp.Choice('dropout_rate', values=[0.3, 0.4, 0.5]))(x)
    x = Dense(len(np.unique(y)), activation='softmax')(x)  # Adjust output based on your number of classes

    new_model = tf.keras.Model(inputs=model.input, outputs=x)

    # Compile the new model with tuned hyperparameters
    new_model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                      loss='sparse_categorical_crossentropy',  # Use sparse categorical if y is encoded
                      metrics=['accuracy'])

    return new_model

# Initialize the Keras Tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3)

# Perform hyperparameter search
tuner.search(train_generator, epochs=10, validation_data=(X_test.reshape(X_test.shape[0], 40, 1, 1), y_test))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Fine-tune the best model
history = best_model.fit(train_generator, epochs=10, validation_data=(X_test.reshape(X_test.shape[0], 40, 1, 1), y_test))

# Save the fine-tuned model
best_model.save('D:/Sound_Recognition/New_Model/Sound_Classifier_CNN_fine.h5')

# Evaluate final performance
loss, accuracy = best_model.evaluate(X_test.reshape(X_test.shape[0], 40, 1, 1), y_test)
print(f'Final accuracy after tuning and fine-tuning: {accuracy * 100:.2f}%')
