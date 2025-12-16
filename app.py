import streamlit as st
import cv2
import numpy as np
import joblib

model = joblib.load("model.pkl")

def extract_features(image_array):
    img = cv2.resize(image_array, (100, 100))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_r, mean_g, mean_b = np.mean(img_rgb[:,:,0]), np.mean(img_rgb[:,:,1]), np.mean(img_rgb[:,:,2])
    std_r, std_g, std_b = np.std(img_rgb[:,:,0]), np.std(img_rgb[:,:,1]), np.std(img_rgb[:,:,2])
    brightness = np.mean(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    return [mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness]

st.title("💧 AI-Powered Water Quality Checker")
st.write("Upload a photo of a water sample to predict its quality (Good / Moderate / Poor).")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Water Sample", use_column_width=True)
    
    features = extract_features(img)
    
    prediction = model.predict([features])[0]
    
    st.subheader("🧠 Predicted Water Quality:")
    if prediction == "Good":
        st.success("✅ Good Quality — Clear and safe-looking water.")
    elif prediction == "Moderate":
        st.warning("⚠️ Moderate Quality — Slightly turbid or cloudy water.")
    else:
        st.error("❌ Poor Quality — Muddy or unsafe-looking water.")
