import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Set page title
st.set_page_config(page_title="Anomaly Detection", layout="centered")

# Load the trained model
model_path = "model/model.h5"
model = tf.keras.models.load_model(model_path)

# Class labels (as used in Teachable Machine)
class_names = ["Normal", "Anomalous"]

# App title
st.title("🔍 Anomaly Detection App")
st.write("Upload an image of a product to check for anomalies using a model trained on Teachable Machine.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Uploaded Image', width=300)


    # Preprocess image for model
    img = image_data.resize((224, 224))  # Change if your model used different size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Show result
    st.subheader("Prediction:")
    st.success(f"🔎 The product is **{predicted_class}**")

    # Optional: Show confidence
    confidence = np.max(prediction) * 100
    st.write(f"Confidence: {confidence:.2f}%")
