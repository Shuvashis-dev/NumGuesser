import pickle
import numpy as np
import streamlit as st
from PIL import Image
import sys

st.title("MNIST Digit Guesser")

with st.spinner("Loading model..."):
    try:
        import numpy.core
        sys.modules['numpy._core'] = numpy.core
    except:
        pass
    
    @st.cache_resource
    def load_model():
        with open("mnist_classifier_model.pkl", "rb") as f:
            return pickle.load(f)
    
    model, scaler = load_model()

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded:
    col1, col2 = st.columns(2)
    
    with col1:
        img = Image.open(uploaded).convert("L")
        st.image(img, width=200)
    
    with col2:
        with st.spinner("Predicting..."):
            small = img.resize((28, 28))
            arr = np.array(small).reshape(1, 784).astype(float) / 255.0
            scaled = scaler.transform(arr)
            pred = model.predict(scaled)[0]
        
        st.write(f"Prediction: {pred}")