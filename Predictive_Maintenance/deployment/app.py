import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="debrupa24/predictive_maintenance_model", filename="predictive_maintenance_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.set_page_config(page_title="Engine Maintenance Prediction", layout="centered")
st.title("Engine Predictive Maintenance Prediction")
st.write("""
This application predicts whether an engine need an immediate maintenace or not
Based on its characteristics such as coolant temperature,lub oil pressure,engine pressure,  etc.
Please enter the app details below to check if specific engine needs a maintenace or not.
""")

# User input
engine_rpm = st.number_input("Engine RPM", min_value=0, value=1000)
lub_oil_pressure = st.number_input("Lub Oil Pressure", min_value=0.0, value=2.5)
fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0, value=2.5)
coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0, value=2.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", min_value=0.0, value=80.0)
coolant_temp = st.number_input("Coolant Temperature", min_value=0.0, value=85.0)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
        "Engine Rpm": engine_rpm,
        "Lub oil pressure": lub_oil_pressure,
        "Coolant pressure": coolant_pressure,
        "Lub oil temp": lub_oil_temp,
        "Coolant temp": coolant_temp
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    if prediction == 1:
            st.success(f"Maintenance Required")
    else:
            st.warning(f"Engine Operating Normally")
