import streamlit as pd
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

# Set professional dashboard metadata
st.set_page_config(page_title="Predictive Maintenance Engine", layout="wide")

st.title("🛡️Automated Engine Health & Predictive Maintenance Platform")
st.markdown("Real-time sensor telemetry processing for vehicle fleet reliability and operational diagnostics.")

@st.cache_resource
def load_production_pipeline():
    """Dynamically fetches the latest verified pipeline artifact from the Model Hub"""
    try:
        # Pulls down from your specific target model repository location
        model_path = hf_hub_download(
            repo_id="amarg7/engine-condition-predictor", 
            filename="model.joblib"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to fetch production model from Hugging Face Hub: {str(e)}")
        return None

pipeline = load_production_pipeline()

# Structural design of the layout using split-view operational categories
st.subheader("📊 Live Sensor Telemetry Input Panels")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ⚙️ Mechanical Speed")
    engine_rpm = st.number_input("Engine rpm (RPM)", min_value=0, max_value=5000, value=750, step=25)

with col2:
    st.markdown("### 💧 Pressure Transducers")
    lub_oil_pressure = st.number_input("Lub oil pressure (bar/kPa)", min_value=0.0, max_value=15.0, value=3.2, step=0.1)
    fuel_pressure = st.number_input("Fuel pressure (bar/kPa)", min_value=0.0, max_value=30.0, value=6.1, step=0.1)
    coolant_pressure = st.number_input("Coolant pressure (bar/kPa)", min_value=0.0, max_value=15.0, value=2.2, step=0.1)

with col3:
    st.markdown("### 🌡️ Thermal Sensors")
    lub_oil_temp = st.number_input("lub oil temp (°C)", min_value=0.0, max_value=150.0, value=77.0, step=0.5)
    coolant_temp = st.number_input("Coolant temp (°C)", min_value=0.0, max_value=150.0, value=79.0, step=0.5)

# Transform operational parameters directly into a standardized runtime DataFrame
input_payload = pd.DataFrame([{
    'engine_rpm': engine_rpm,
    'lub_oil_pressure': lub_oil_pressure,
    'fuel_pressure': fuel_pressure,
    'coolant_pressure': coolant_pressure,
    'lub_oil_temp': lub_oil_temp,
    'coolant_temp': coolant_temp
}])

st.markdown("---")

if st.button("🚀 Process Telemetry Diagnostics", use_container_width=True):
    if pipeline is not None:
        # Generate model class probability arrays
        probabilities = pipeline.predict_proba(input_payload)[:, 1]
        raw_prob = probabilities[0]
        
        # Apply the calibrated cost-aware business classification threshold
        BUSINESS_THRESHOLD = 0.45
        prediction = 1 if raw_prob >= BUSINESS_THRESHOLD else 0
        
        # Interface rendering based on structural classification outcomes
        if prediction == 1:
            st.error(f"🚨 **CRITICAL RISK DETECTED: MAINTENANCE MANDATORY** (Failure Likelihood: {raw_prob:.2%})")
            st.markdown("""
            **Recommended Field Actions:**
            1. Flag asset immediately in Central Fleet Control to isolate from route planning.
            2. Inspect mechanical fuel delivery networks for high-pressure stress build-up.
            3. Conduct a targeted inspection of lubrication pump flow volume.
            """)
        else:
            st.success(f"✅ **ENGINE RUNNING WITHIN NORMAL OPERATIONAL LIMITS** (Failure Likelihood: {raw_prob:.2%})")
            st.markdown("**Status Notice:** Telemetry signature matches baseline wear metrics. Continue with standard scheduling.")
    else:
        st.warning("Diagnostics unavailable: Model pipeline asset unverified.")
