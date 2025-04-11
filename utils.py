from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

def preprocess_input(user_input):
    values = np.array([float(x.strip()) for x in user_input.split(",")]).reshape(1, -1)
    return scaler.transform(values)

def predict_intrusion(preprocessed_input):
    prediction = model.predict(preprocessed_input)
    return prediction
