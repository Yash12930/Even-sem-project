import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import zipfile
import os

# Define a function to load and preprocess the image
def load_and_prep_image(image_file):
    img = Image.open(image_file).resize((32, 32))
    img = np.array(img) / 255.0
    return img

# Load the pre-trained model
model = tf.keras.models.load_model('cifake_classifier.h5')

# Streamlit app
st.title('Image Classification: Real or Fake')

st.write('Upload an image to classify it as Real or Fake')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = load_and_prep_image(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)

    if predicted_class == 0:
        st.write("This image is classified as: Real")
    else:
        st.write("This image is classified as: Fake")
