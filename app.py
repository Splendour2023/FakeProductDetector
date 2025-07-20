from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('./fake_product_model.h5')  # Adjust if your model path is different

# Route for home page
@app.route('/')
def home():
    return '''
        <h1>Fake Product Detector</h1>
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <p>Upload Product Image:</p>
            <input type="file" name="file" accept="image/*">
            <br><br>
            <input type="submit" value="Predict">
        </form>
    '''

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part in the request.'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file.'

    # Save the uploaded file
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    # Preprocess the image
    img = image.load_img(filepath, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    prediction = model.predict(img_tensor)[0][0]

    label = "Real Product ✅" if prediction >= 0.5 else "Fake Product ❌"
    return f"<h2>Prediction: {label}</h2><p>Confidence Score: {prediction:.2f}</p>"
    
if __name__ == '__main__':
    app.run(debug=True)

