import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel

# Load the scaler
loaded_scaler = joblib.load('standard_scaler.pkl')

# Load the saved autoencoder model
loaded_autoencoder = load_model("autoencoder_model.h5")

# Load Featurewise Threshold
feature_wise_threshold = joblib.load("feature_wise_threshold.pkl")

# Threshold for reconstruction loss
threshold = 0.0005

# Create a FastAPI instance
app = FastAPI(title='DC Motor Fault Detection API')

# Define the input data model using Pydantic
class InputData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return "Congratulations! Your API is working as expected"

# Define a predict endpoint
@app.post("/predict")
def predict(data: InputData):
    """
    Endpoint to make predictions based on input features.

    Parameters:
    - data: InputData, Pydantic model representing input features.

    Returns:
    - dict: Dictionary containing reconstruction loss and feature-wise reconstruction.
    """
    # Access the input data
    input_values = data.features

    # Transform the input data using the loaded scaler
    x = loaded_scaler.transform([input_values])

    # Predict using the loaded autoencoder
    x_recon = loaded_autoencoder.predict(x)

    # Calculate reconstruction loss
    reconstruction_loss = np.mean((x - x_recon) ** 2, axis=1) / threshold

    # Calculate feature-wise reconstruction
    feature_wise_recon = np.abs(x - x_recon) / feature_wise_threshold

    # Return the results
    return {"reconstruction_loss": reconstruction_loss.tolist(), "feature_wise_recon": feature_wise_recon[0].tolist()}
