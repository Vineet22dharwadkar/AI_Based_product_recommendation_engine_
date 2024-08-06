import os
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
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

def load_pickle_model(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"File not found: {file_path}")
        return None

feature_list_resnet = np.array(pickle.load(open("Models/featurevector_resnet50.pkl", "rb")))
filenames_resnet = pickle.load(open("Models/filenames_resnet50.pkl", "rb"))
feature_list_vgg = np.array(pickle.load(open("Models/featurevector_vgg16.pkl", "rb")))
filenames_vgg = pickle.load(open("Models/filenames_vgg16.pkl", "rb"))
feature_list_inception = np.array(pickle.load(open("Models/featurevector_inceptionv3.pkl", "rb")))
filenames_inception = pickle.load(open("Models/filenames_inceptionv3.pkl", "rb"))
feature_list_efficientnet = np.array(pickle.load(open("Models/featurevector_efficientnetb0.pkl", "rb")))
filenames_efficientnet = pickle.load(open("Models/filenames_efficientnetb0.pkl", "rb"))

knn_model = load_pickle_model('Models/knn_model.pkl')
svm_model = load_pickle_model('Models/svm_model.pkl')
rf_model = load_pickle_model('Models/rf_model.pkl')
dt_model = load_pickle_model('Models/dt_model.pkl')

models = {
    "resnet50": {
        "model": tf.keras.Sequential([ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), GlobalMaxPooling2D()]),
        "preprocess": resnet_preprocess,
        "feature_list": feature_list_resnet,
        "filenames": filenames_resnet
    },
    "vgg16": {
        "model": tf.keras.Sequential([VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), GlobalMaxPooling2D()]),
        "preprocess": vgg_preprocess,
        "feature_list": feature_list_vgg,
        "filenames": filenames_vgg
    },
    "inceptionv3": {
        "model": tf.keras.Sequential([InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), GlobalMaxPooling2D()]),
        "preprocess": inception_preprocess,
        "feature_list": feature_list_inception,
        "filenames": filenames_inception
    },
    "efficientnetb0": {
        "model": tf.keras.Sequential([EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3)), GlobalMaxPooling2D()]),
        "preprocess": efficientnet_preprocess,
        "feature_list": feature_list_efficientnet,
        "filenames": filenames_efficientnet
    }
}

def extract_feature(img_path, model, preprocess_input):
    print(f"Extracting features from: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {img_path}")
    img = cv2.resize(img, (224, 224))
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
    if "file" not in request.files or "model" not in request.form:
        return jsonify({"error": "No file or model part"}), 400
    file = request.files["file"]
    model_name = request.form["model"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if model_name not in models:
        return jsonify({"error": "Invalid model name"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        model_info = models[model_name]
        features = extract_feature(file_path, model_info["model"], model_info["preprocess"])
        indices = recommend(features, model_info["feature_list"])
        similar_images = [model_info["filenames"][idx] for idx in indices[0]]
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
    model_name = data["model"]
    if model_name not in models:
        return jsonify({"error": "Invalid model name"}), 400
    model_info = models[model_name]
    features = extract_feature(image_path, model_info["model"], model_info["preprocess"])
    indices = recommend(features, model_info["feature_list"])
    similar_images = [model_info["filenames"][idx] for idx in indices[0]]
    return jsonify(similar_images)

@app.route("/uploads/<path:filename>")
def send_uploaded_image(filename):
    return send_from_directory("uploads", filename)

@app.route("/Dataset/<path:filename>")
def send_dataset_image(filename):
    return send_from_directory("Dataset", filename)

# New endpoints for the trained models
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        features = extract_feature(file_path, models["vgg16"]["model"], vgg_preprocess)  # Using VGG16 for feature extraction

        predictions = {}

        if knn_model:
            knn_prediction = knn_model.predict([features])[0]
            predictions["knn"] = knn_prediction
        else:
            predictions["knn"] = "Model not found"

        if svm_model:
            svm_prediction = svm_model.predict([features])[0]
            predictions["svm"] = svm_prediction
        else:
            predictions["svm"] = "Model not found"

        if rf_model:
            rf_prediction = rf_model.predict([features])[0]
            predictions["random_forest"] = rf_prediction
        else:
            predictions["random_forest"] = "Model not found"

        if dt_model:
            dt_prediction = dt_model.predict([features])[0]
            predictions["decision_tree"] = dt_prediction
        else:
            predictions["decision_tree"] = "Model not found"

        return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)
