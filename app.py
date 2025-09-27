# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# === CONFIG ===
MODEL_FILENAME = "xgb_final_model.joblib"   # Rename your uploaded file to this and keep in same folder as app.py
FEATURES = ["Rainfall", "Fertilizer", "Temperature"]  # Update to match your model training features

# === LOAD MODEL ===
MODEL_PATH = MODEL_FILENAME
model = None
try:
    model = joblib.load(MODEL_PATH)
    load_msg = f"Loaded model: {MODEL_FILENAME}"
except Exception as e:
    model = None
    load_msg = f"Model not loaded: {e}"

# === STREAMLIT UI ===
st.title("🌾 Crop Yield Prediction (XGBoost)")
st.caption(load_msg)

st.header("Enter input features")

inputs = []
cols = st.columns(2)
for i, feat in enumerate(FEATURES):
    if i % 2 == 0:
        val = cols[0].number_input(feat, value=0.0)
    else:
        val = cols[1].number_input(feat, value=0.0)
    inputs.append(val)

X_new = pd.DataFrame([inputs], columns=FEATURES)

st.write("Input preview:")
st.dataframe(X_new)

if st.button("Predict Yield"):
    if model is None:
        st.error("❌ Model not loaded. Make sure xgb_final_model.joblib is in the same folder as app.py.")
    else:
        try:
            pred = model.predict(X_new)[0]
            st.success(f"✅ Predicted Yield: {pred:.4f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write("Check terminal for full traceback.")
