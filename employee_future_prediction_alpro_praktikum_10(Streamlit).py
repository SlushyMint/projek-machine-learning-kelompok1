import joblib
import streamlit as st
import pandas as pd

# Load Model and Encoder
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')  

st.title("Prediksi Retensi Karyawan")

# Input Dari Pengguna
with st.form("employee_form"):
    Education = st.selectbox("Education (0: Bachelor, 1: Master)", ["0", "1"]) 
    JoiningYear = st.number_input('Joining Year', min_value=2000, max_value=2024, value=2023, step=1)
    City = st.selectbox('City (0: Bangalore, 1: New Delhi, 2: Pune)', ["0", "1", "2"])
    PaymentTier = st.selectbox('Payment Tier (0: Tier 1, 1: Tier 2, 2: Tier 3)', ["0", "1", "2"])
    Age = st.number_input('Age', min_value=18, max_value=65, value=25, step=1)
    Gender = st.selectbox('Gender (0: Female, 1: Male)', ["0", "1"])
    EverBenched = st.selectbox('Ever Benched (0: No, 1: Yes)', ["0", "1"]) 
    ExperienceInCurrentDomain = st.number_input('Experience in Current Domain', min_value=0, max_value=20, value=0, step=1)
    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        input_data = {
            'Education': Education,
            'JoiningYear': JoiningYear,
            'City': City,
            'PaymentTier': PaymentTier,
            'Age': Age,
            'Gender': Gender,
            'EverBenched': EverBenched,
            'ExperienceInCurrentDomain': ExperienceInCurrentDomain
        }

        try:
            prediction_labels = {
                0: "Not Leaving",
                1: "Leaving"
            }
            input_df = pd.DataFrame([input_data])
            input_reshaped = input_df.values.reshape(1, -1)
            prediction = model.predict(input_reshaped)
            prediction_labels = prediction_labels[prediction[0]]
            st.write(f"Prediksi: {prediction_labels}")

            
        except Exception as e:
            st.write(f"Error: {str(e)}")