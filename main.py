import streamlit as st
from PIL import Image
import io
import base64
import os
from app.utils.app import predict_parkinsons

# Configure the page
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="centered",
)

# Add custom CSS for improved styling
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e2f;
        color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    .stText, .stMarkdown, .stButton, .stSelectbox, .stFileUploader, .stSlider, .stRadio, .stCheckbox, .stDownloadButton {
        color: #f5f5f5;
    }
    .stTitle {
        color: #00FF7F;
    }
    .stSidebar {
        background-color: #2c2c3c;
        color: #f5f5f5;
    }
    .stSidebar .stImage {
        background-color: #2c2c3c;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader {
        background-color: #333333;
        border-radius: 5px;
    }
    .stFileUploader input {
        color: #f5f5f5;
    }
    .stTextInput>input {
        color: #f5f5f5;
        background-color: #444444;
        border: none;
    }
    .prediction-positive {
        text-align: center;
        color: #FF6F61;
        font-size: 24px;
        font-weight: bold;
    }
    .prediction-negative {
        text-align: center;
        color: #32CD32;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Helper function to encode images in Base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Paths for assets
parkinson_image_path = "assets/images/parkinson.jpg"
audio_icon_path = "assets/images/audio_icon.png"
image_icon_path = "assets/images/image_icon.png"

# Load images
parkinson_image = Image.open(parkinson_image_path)
audio_icon_base64 = encode_image(audio_icon_path)
image_icon_base64 = encode_image(image_icon_path)


# Sidebar with information
with st.sidebar:
    st.image(parkinson_image, caption="Parkinson's Awareness", use_container_width=True)
    st.markdown("### About Parkinson's Disease")
    st.markdown(
        '<p style="color:white;">Parkinson\'s disease is a progressive nervous system disorder that affects movement. '
        'Early detection and diagnosis can help manage symptoms more effectively.</p>',
        unsafe_allow_html=True 
    )


# Main title and description
st.title("🧠 Parkinson's Disease Prediction")
st.markdown(
    "<div style='text-align: center; font-size: 18px; color: black;'>"
    "Upload an audio file and a drawing image to check for Parkinson's Disease."
    "</div>",
    unsafe_allow_html=True,
)

# Input form with icons
st.markdown(
    f"<div style='display: flex; align-items: center; margin-bottom: 10px;'>"
    f"<img src='data:image/png;base64,{audio_icon_base64}' alt='Audio Icon' style='width: 24px; margin-right: 8px;'>"
    f"<label for='audio_file' style='color: #f5f5f5;'>Audio File</label>"
    "</div>",
    unsafe_allow_html=True,
)
audio_file = st.file_uploader(
    "Upload an audio file (.wav)",
    type=["wav"],
    label_visibility="collapsed",
    accept_multiple_files=False,
    help="Upload an audio recording in WAV format."
)

st.markdown(
    f"<div style='display: flex; align-items: center; margin-bottom: 10px;'>"
    f"<img src='data:image/png;base64,{image_icon_base64}' alt='Image Icon' style='width: 24px; margin-right: 8px;'>"
    f"<label for='image_file' style='color: #f5f5f5;'>Drawing Image</label>"
    "</div>",
    unsafe_allow_html=True,
)
image_file = st.file_uploader(
    "Upload a drawing image (.png, .jpg)",
    type=["png", "jpg"],
    label_visibility="collapsed",
    accept_multiple_files=False,
    help="Upload a drawing image in PNG or JPG format."
)

# Prediction button
if st.button("Predict"):
    if audio_file and image_file:
        try:
            # Save uploaded files to temporary paths
            audio_path = f"temp_{audio_file.name}"
            image_path = f"temp_{image_file.name}"

            with open(audio_path, "wb") as f:
                f.write(audio_file.getvalue())

            with open(image_path, "wb") as f:
                f.write(image_file.getvalue())

            # Call the prediction function
            result = predict_parkinsons(audio_path, image_path)

            if result['prediction'] == "Positive":
                st.markdown(
                    f"<div class='prediction-positive'>"
                    f"Prediction: You may have Parkinson's Disease."
                    f"<p>Confidence: {result['confidence']:.2f}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='prediction-negative'>"
                    f"Prediction: You are unlikely to have Parkinson's Disease."
                    f"<p>Confidence: {result['confidence']:.2f}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Display audio player
            st.audio(audio_file)
            # Display image
            image = Image.open(io.BytesIO(image_file.getvalue()))
            st.image(image, caption="Uploaded Drawing", width=300)

        finally:
            # Delete the temporary files after prediction
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                st.error(f"Failed to delete temporary files: {e}")

    else:
        st.error("Please upload both audio and image files.")

