import json
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱")

working_dir = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def download_and_load_model():
    """Download model from Google Drive if not exists, then load it"""

    model_path = f"{working_dir}/plant_disease_prediction_model.h5"

    # Check if model already downloaded
    if not os.path.exists(model_path):
        st.info("🔄 Downloading model from Google Drive... This will take a moment (one-time only).")

        # Your Google Drive file ID
        file_id = '1aSUx8N20oPFc0pbdyBsZXet_YwHWpSz8'
        url = f'https://drive.google.com/uc?id={file_id}'

        try:
            # Download the model
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {str(e)}")
            st.stop()

    # Load the model
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


# Load the pre-trained model
model = download_and_load_model()

# Loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('🌱 Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            with st.spinner('🔍 Analyzing...'):
                prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
