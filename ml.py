import streamlit as st
import pickle
import numpy as np
import cv2
from numpy.linalg import norm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow as tf
import os

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_model.trainable = False
vgg_model = tf.keras.Sequential([vgg_model, GlobalMaxPooling2D()])

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

knn_model = load_model('Models/knn_model.pkl')
svm_model = load_model('Models/svm_model.pkl')
rf_model = load_model('Models/rf_model.pkl')
dt_model = load_model('Models/dt_model.pkl')

def extract_feature(img_path, model):
    try:
        img = cv2.imread(img_path)
        if img is None:
            st.error(f"Failed to load image: {img_path}")
            return None
        img = cv2.resize(img, (224, 224))
        img = np.array(img)
        expand_img = np.expand_dims(img, axis=0)
        pre_img = preprocess_input(expand_img)
        result = model.predict(pre_img).flatten()
        normalized = result / norm(result)
        return normalized
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def display_images(image_paths):
    cols = st.columns(5)
    for idx, image_path in enumerate(image_paths):
        try:
            cols[idx % 5].image(image_path, use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image {image_path}: {e}")

st.title('Image Classification with ML Models')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    features = extract_feature("uploaded_image.jpg", vgg_model)

    if features is not None:
        knn_prediction = knn_model.predict([features])[0] if knn_model else "Model not found"
        svm_prediction = svm_model.predict([features])[0] if svm_model else "Model not found"
        rf_prediction = rf_model.predict([features])[0] if rf_model else "Model not found"
        dt_prediction = dt_model.predict([features])[0] if dt_model else "Model not found"

        st.write("Predictions:")
        st.write(f"KNN: {knn_prediction}")
        st.write(f"SVM: {svm_prediction}")
        st.write(f"Random Forest: {rf_prediction}")
        st.write(f"Decision Tree: {dt_prediction}")

        image_dir = 'Dataset' 
        knn_image_paths = [os.path.join(image_dir, f"{knn_prediction}")]
        svm_image_paths = [os.path.join(image_dir, f"{svm_prediction}")]
        rf_image_paths = [os.path.join(image_dir, f"{rf_prediction}")]
        dt_image_paths = [os.path.join(image_dir, f"{dt_prediction}")]

        st.write("KNN Predicted Image:")
        display_images(knn_image_paths)

        st.write("SVM Predicted Image:")
        display_images(svm_image_paths)

        st.write("Random Forest Predicted Image:")
        display_images(rf_image_paths)

        st.write("Decision Tree Predicted Image:")
        display_images(dt_image_paths)

