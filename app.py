# app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title and description
st.title("Iris Flower Type Predictor")
st.write("""
This app predicts the type of Iris flower based on the measurements you provide below.  
Please adjust the sliders to input flower features, then click **Predict** to see the result.
""")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Prepare input features
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
scaled_features = scaler.transform(features)

# Predict when the button is clicked
if st.button("Predict"):
    prediction = model.predict(scaled_features)
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_class = class_names[prediction[0]]
    
    st.subheader("Prediction Result")
    st.write(f"The predicted Iris flower type is: **{predicted_class}**")