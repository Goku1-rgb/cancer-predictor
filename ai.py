import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf

# Load data
dataset = pd.read_csv("cancer.csv")
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Save scaler for later use in inference
joblib.dump(scaler, "../../Downloads/scaler.save")

# Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(x_train_scaled.shape[1],)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(x_train_scaled, y_train, epochs=10, validation_split=0.2)

# Optionally evaluate on test set
test_loss, test_acc = model.evaluate(x_test_scaled, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Save model
model.save("cancer_model", save_format="tf")
