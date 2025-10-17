import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os
import cv2


# Custom deserialization to handle the InputLayer issue
def load_model_with_fix(model_path):
    """Load model with custom object handling for compatibility"""
    try:
        # Try normal loading first
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Attempting alternative loading method...")

        # Alternative: Load weights only
        try:
            # Recreate the model architecture
            model = keras.Sequential([
                keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['recall'])

            # Load weights from the saved model
            model.load_weights(model_path)
            return model
        except:
            st.error("Could not load model. Please retrain and save the model using the fixed script.")
            return None


# Set working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'modèle_ia_projet.keras')

# Check if model exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# Load the pre-trained model
with st.spinner('Loading model...'):
    model = load_model_with_fix(model_path)

if model is None:
    st.error("Failed to load model. Please check the model file.")
    st.stop()

st.success("Model loaded successfully!")

# Define class labels for your binary classification
class_names = ['Normal', 'Tuberculosis']


# Function to preprocess the uploaded image
def preprocess_image(image):
    # Open and convert PIL Image
    img = Image.open(image)

    # Convert PIL Image to numpy array
    img_array = np.array(img)

    # Convert to BGR (OpenCV format) if RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif len(img_array.shape) == 2:
        # Already grayscale
        img_array = img_array

    # Resize to 128x128 (same as training)
    img_resized = cv2.resize(img_array, (128, 128))

    # Convert to grayscale if needed
    if len(img_resized.shape) == 3:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_resized

    # Normalize to [0, 1]
    img_normalized = img_gray.astype('float32') / 255.0

    # Reshape to match model input: (1, 128, 128, 1)
    img_final = img_normalized.reshape(1, 128, 128, 1)

    return img_final


# Streamlit App
st.title('🫁 Tuberculosis Detection from Chest X-Ray')
st.write('Upload a chest X-ray image to detect Tuberculosis')

uploaded_image = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        resized_img = image.resize((300, 300))
        st.image(resized_img)

    with col2:
        if st.button('🔍 Classify'):
            with st.spinner('Analyzing...'):
                try:
                    # Preprocess the uploaded image
                    img_array = preprocess_image(uploaded_image)

                    # Make a prediction using the pre-trained model
                    result = model.predict(img_array, verbose=0)

                    # Get prediction probability
                    prediction_prob = result[0][0]

                    # Binary classification: threshold at 0.5
                    predicted_class = 1 if prediction_prob > 0.5 else 0
                    prediction = class_names[predicted_class]

                    # Display results
                    st.subheader("Results")
                    if predicted_class == 1:
                        st.error(f'⚠️ Prediction: **{prediction}**')
                        st.write(f'Confidence: {prediction_prob:.2%}')
                    else:
                        st.success(f'✅ Prediction: **{prediction}**')
                        st.write(f'Confidence: {(1 - prediction_prob):.2%}')

                    # Show probability bar
                    st.progress(float(prediction_prob))

                    st.info(
                        '**Note:** This is a machine learning prediction and should not be used as a medical diagnosis. Please consult a healthcare professional.')

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
    This application uses a Convolutional Neural Network (CNN) 
    to detect Tuberculosis from chest X-ray images.

    **Model Details:**
    - Input size: 128x128 grayscale images
    - Architecture: CNN with Conv2D, MaxPooling, and Dense layers
    - Binary classification: Normal vs Tuberculosis

    **How to use:**
    1. Upload a chest X-ray image (JPG, JPEG, or PNG)
    2. Click the 'Classify' button
    3. View the prediction results
    """)

    st.header("Disclaimer")
    st.warning("""
    This tool is for educational purposes only. 
    It should NOT be used for actual medical diagnosis. 
    Always consult qualified healthcare professionals for medical advice.
    """)