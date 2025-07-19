import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model



scaler = StandardScaler()
joblib.load("scaler.save")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


dataset = pd.read_csv("cancer.csv")

x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

model=load_model("cancer_model.h5")
st.title("Cancer Diagnosis Predictor")
st.write("Enter the patient's data to get a prediction.")
input_data = []
feature_names = list(x.columns)


for feature in feature_names:
    val = st.number_input(f"Enter the value for {feature}:")
    input_data.append(val)



if st.button("Predict"):
    x_input = np.array(input_data).reshape(1, -1)
    x_scaled = scaler.transform(x_input)  # Use pre-fitted scaler

    prediction = model.predict(x_scaled)[0][0]
    diagnosis = "Malignant" if prediction > 0.5 else "Benign"

    st.success(f"Prediction: {diagnosis} ({prediction:.2f})")


