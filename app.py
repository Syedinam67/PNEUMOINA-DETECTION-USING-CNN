import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load your trained TensorFlow model
model = tf.keras.models.load_model("trained.h5")

# Allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def predict_image(image_path):
    # Preprocess image
    img = image.load_img(image_path, target_size=(300, 300))  # Adjust size as per model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize if required

    # Make prediction
    predictions = model.predict(img_array)
    probability = predictions[0][0]  # Assuming single-output binary classification
    threshold = 0.5  # Adjust if needed
    return "Pneumonia" if probability >= threshold else "Normal"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file is present and valid
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform prediction
            prediction = predict_image(filepath)
            return render_template('ind.html', prediction=prediction, image_path=url_for('static', filename=f'uploads/{filename}'))

    return render_template('ind.html', prediction=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)


