import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="plaban95/Predictive_Maintenance", filename="Best_Predictive_Maintenance_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Predictive Maintenance
st.title("Predictive Maintenance App")
st.write("""
This application predicts the likelihood of an engine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
engine_rpm = st.number_input("Engine rpm (RPM)", min_value=10.0, max_value=10000.0, value=0, step=1)
lub_oil_pressure = st.number_input("Lub oil pressure (Kpa)", min_value=1.0, max_value=100.0, value=0.0, step=0.1)
fuel_pressure = st.number_input("Fuel pressure (Kpa)", min_value=1.0, max_value=100.0, value=0.0, step=0.1)
coolant_pressure = st.number_input("Coolant pressure (Kpa)", min_value=1.0, max_value=100.0, value=0.0, step=0.1)
lub_oil_temp = st.number_input("lub oil temp (C)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
coolant_temp = st.number_input("Coolant temp (C)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant pressure': coolant_pressure,
    'Coolant temp': coolant_temp,
    'Fuel pressure' : fuel_pressure
}])


if st.button("Predict condition"):
    prediction = model.predict(input_data)[0]
    result = "Engine Failure" if prediction == 1 else "Engine Ok"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
