import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Turn off ONEDNN optimizations for numerical consistency
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load dataset and prepare scaler
dataset = pd.read_csv("cancer.csv")
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

scaler = StandardScaler()
scaler.fit(x)  # ✅ Fit scaler from your dataset

# Load model
try:
    model = load_model("cancer_model.h5", compile =True)
    print("Model loaded")
except Exception as e:
    print("Error loading model:")
    print(e)# Streamlit UI
st.title("Cancer Diagnosis Predictor")
st.write("Enter the patient's data to get a prediction.")

input_data = []
feature_names = list(x.columns)

for feature in feature_names:
    val = st.number_input(f"Enter the value for {feature}:")
    input_data.append(val)

if st.button("Predict"):
    x_input = np.array(input_data).reshape(1, -1)
    x_scaled = scaler.transform(x_input)  # ✅ Use fitted scaler

    prediction = model.predict(x_scaled)[0][0]
    diagnosis = "Malignant" if prediction > 0.5 else "Benign"

    st.success(f"Prediction: {diagnosis} ({prediction:.2f})")
print("Tensorflow version:", tf.__version__)
print("Files in current directory:", os.listdir())