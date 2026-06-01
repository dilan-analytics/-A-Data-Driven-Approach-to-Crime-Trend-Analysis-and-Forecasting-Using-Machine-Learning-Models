import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('crime_prediction_model.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(
    page_title="Crime Forecasting System",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Crime Forecasting System")
st.markdown("Enter current crime rates to predict next year's violent crime rate.")

st.divider()

st.subheader("Input Current Crime Data")

col1, col2, col3 = st.columns(3)

with col1:
    violent_rate = st.number_input(
        "Violent Crime Rate",
        min_value=0.0,
        step=0.1,
        help="Enter current violent crime rate"
    )

with col2:
    property_rate = st.number_input(
        "Property Crime Rate",
        min_value=0.0,
        step=0.1,
        help="Enter current property crime rate"
    )

with col3:
    firearm_rate = st.number_input(
        "Firearm Crime Rate",
        min_value=0.0,
        step=0.1,
        help="Enter current firearm-related crime rate"
    )

st.divider()

if st.button("🔮 Predict Crime Rate", use_container_width=True):

    input_data = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    if 'Violent_Lag1' in feature_names:
        input_data['Violent_Lag1'] = violent_rate
    if 'Property_Lag1' in feature_names:
        input_data['Property_Lag1'] = property_rate
    if 'Firearm_Lag1' in feature_names:
        input_data['Firearm_Lag1'] = firearm_rate

    prediction = model.predict(input_data)[0]

    st.success("Prediction Completed 🎯")

    st.metric(
        label="Forecasted Violent Crime Rate (Next Year)",
        value=f"{prediction:.2f}"
    )

    if prediction < 50:
        st.info("Low risk level predicted.")
    elif prediction < 100:
        st.warning("Moderate risk level predicted.")
    else:
        st.error("High risk level predicted. Attention required.")

st.divider()
st.caption("Crime Forecasting ML System | Built with Streamlit")