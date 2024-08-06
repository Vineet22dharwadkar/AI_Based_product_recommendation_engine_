import os
import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
import cv2
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load and prepare the VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_model.trainable = False
vgg_model = tf.keras.Sequential([vgg_model, GlobalMaxPooling2D()])

# Load trained models
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

knn_model = load_model('knn_model.pkl')
svm_model = load_model('svm_model.pkl')
rf_model = load_model('rf_model.pkl')
dt_model = load_model('dt_model.pkl')

UPLOAD_FOLDER = 'uploads'
DATASET_FOLDER = 'Dataset'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_feature(img_path, model):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.resize(img, (224, 224))
        img = np.array(img)
        expand_img = np.expand_dims(img, axis=0)
        pre_img = preprocess_input(expand_img)
        result = model.predict(pre_img).flatten()
        normalized = result / norm(result)
        return normalized
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        features = extract_feature(file_path, vgg_model)
        if features is None:
            return jsonify({'error': 'Error extracting features from image'}), 500

        knn_prediction = knn_model.predict([features])[0] if knn_model else "Model not found"
        svm_prediction = svm_model.predict([features])[0] if svm_model else "Model not found"
        rf_prediction = rf_model.predict([features])[0] if rf_model else "Model not found"
        dt_prediction = dt_model.predict([features])[0] if dt_model else "Model not found"

        return jsonify({
            'knn': knn_prediction,
            'svm': svm_prediction,
            'rf': rf_prediction,
            'dt': dt_prediction
        })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/dataset/<filename>')
def dataset_file(filename):
    return send_from_directory(DATASET_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
