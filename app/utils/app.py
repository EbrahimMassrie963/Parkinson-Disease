# app.py
import os
import joblib
import numpy as np
import librosa
import cv2
import numpy as np
from skimage import feature
from sklearn.preprocessing import StandardScaler

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "Rf_edit.pkl"



def extract_voice_features(audio_file_path):
    """
    Extracts MFCC features from an audio file and applies scaling.

    Parameters:
        audio_file_path (str): Path to the audio file.

    Returns:
        np.ndarray: Scaled MFCC features as a 2D array.
    """
    samples, sample_rate = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=9)
    features_mean = [mfcc.mean() for mfcc in mfccs]
    voice_features = np.array(features_mean)
    scaler = StandardScaler()
    return scaler.fit_transform(voice_features.reshape(-1, 1)).reshape(1, -1)

def extract_img_features(image_file_path):
    """
    Extracts HOG features from an image file.

    Parameters:
        image_file_path (str): Path to the image file.

    Returns:
        np.ndarray: HOG features as a 2D array.
    """
    img = cv2.imread(image_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (250, 250))
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    hog_features = feature.hog(img, orientations=9,
                               pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                               transform_sqrt=True, block_norm="L1")
    return hog_features.reshape(1, -1)


try:
    if not MODEL_PATH:
        raise ValueError("Model path is not set.")
    loaded_model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

def predict_parkinsons(audio_path, image_path):
    """
    Predicts Parkinson's disease using audio and image features.

    Parameters:
        audio_path (str): Path to the audio file.
        image_path (str): Path to the image file.

    Returns:
        dict: Prediction result containing the prediction label and confidence.
    """
    try:
        # Extract features
        voice_features = extract_voice_features(audio_path)
        img_features = extract_img_features(image_path)

        # Concatenate features
        model_input = np.concatenate((voice_features, img_features), axis=1)

        # Make prediction
        prediction = loaded_model.predict(model_input)[0]
        confidence = max(loaded_model.predict_proba(model_input)[0])

        return {
            "prediction": "Positive" if prediction == 1 else "Negative",
            "confidence": confidence
        }

    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")


