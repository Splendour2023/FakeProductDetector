import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('model/fake_product_detector.h5')

# Path to the test image
img_path = 'test_images/sample.jpg'  # <--- Change this to your test image

# Load and preprocess the image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
class_names = ['fake', 'real']
predicted_class = class_names[int(prediction[0][0] > 0.5)]

print(f'✅ Prediction: This product is {predicted_class.upper()}!')
