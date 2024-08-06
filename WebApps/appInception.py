import os
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_PATH"] = 16 * 1024 * 1024

feature_list = np.array(pickle.load(open("featurevector_inceptionv3.pkl", "rb")))
filenames = pickle.load(open("filenames_inceptionv3.pkl", "rb"))

model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

def extract_feature(img_path, model):
    print(f"Extracting features from: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {img_path}")
    img = cv2.resize(img, (299, 299))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

def get_random_images(folder_path, num_images):
    all_images = os.listdir(folder_path)
    random_images = random.sample(all_images, min(num_images, len(all_images)))
    return random_images

@app.route("/upload", methods=["POST"])
def upload_image():
    print("Received request to upload image")
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        features = extract_feature(file_path, model)
        indices = recommend(features, feature_list)
        similar_images = [filenames[idx] for idx in indices[0]]
        return jsonify(similar_images)

@app.route("/random_images", methods=["GET"])
def random_images():
    image_folder = "uploads"
    images = get_random_images(image_folder, 20)
    return jsonify(images)

@app.route("/similar_images", methods=["POST"])
def similar_images():
    data = request.get_json()
    image_path = os.path.join("uploads", data["image"])
    features = extract_feature(image_path, model)
    indices = recommend(features, feature_list)
    similar_images = [filenames[idx] for idx in indices[0]]
    return jsonify(similar_images)

@app.route("/uploads/<path:filename>")
def send_uploaded_image(filename):
    return send_from_directory("uploads", filename)

@app.route("/Dataset/<path:filename>")
def send_dataset_image(filename):
    return send_from_directory("Dataset", filename)

if __name__ == "__main__":
    app.run(debug=True)
