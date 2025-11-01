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
