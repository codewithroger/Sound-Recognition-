import tensorflow as tf
import os  # Import os module to work with directories

# Load the Keras model
model = tf.keras.models.load_model('D:/Sound_Recognition/Sound_Model/Sound_Classifier.h5')

# Convert the model to TensorFlow Lite format with optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable optimization for size and speed
tflite_quant_model = converter.convert()

# Specify the path where you want to save the converted model
save_dir = 'D:/Sound_Recognition/Tflite_Models'
save_path = os.path.join(save_dir, 'Sound_Recognition.tflite')

# Check if the directory exists, and create it if not
os.makedirs(save_dir, exist_ok=True)

# Save the converted and optimized model
with open(save_path, 'wb') as f:
    f.write(tflite_quant_model)  # Saving to the specified path

print(f"Model converted and saved as {save_path}")
