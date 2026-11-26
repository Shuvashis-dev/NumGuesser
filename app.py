from pathlib import Path
import pickle

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


MODEL_PATH = Path(__file__).resolve().parent / "mnist_classifier_model.pkl"


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


st.title("MNIST Digit Recognition")

model, scaler = load_model()
uploaded_file = st.file_uploader(
    "Upload a digit",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded image", width=200)

    image = ImageOps.invert(image).resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32).reshape(1, 784)
    scaled_image = scaler.transform(image_array)
    prediction = model.predict(scaled_image)

    st.success(f"Predicted Number: {prediction[0]}")
