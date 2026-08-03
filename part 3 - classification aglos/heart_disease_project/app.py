
import streamlit as st
import pandas as pd
import joblib
# Logistic Regression model thats was best 
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')


st.title("Heart Disease Prediction App")
st.markdown("This app predicts the presence of heart disease based on user input features.")


age = st.slider("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", ['Male', 'Female'])
chest_pain_type = st.selectbox("Chest Pain Type", ['TA', 'ATA', 'NAP', 'ASY'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_BS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['Yes', 'No'])
resting_ecg = st.selectbox("Resting Electrocardiographic Results", ['Normal', 'ST', 'LVH'])
max_hr = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
oldpeak = st.slider("Oldpeak (ST depression induced by exercise relative to rest)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox("Slope of the peak exercise ST segment", ['Up', 'Flat', 'Down'])


if st.button("Predict"):
    raw_input = {
        'Age' : age,
        'RestingBP' : resting_bp,
        'Cholesterol' : cholesterol,
        'FastingBS' : 1 if fasting_BS == 'Yes' else 0,
        'MaxHR' : max_hr,
        'Oldpeak' : oldpeak,
        'Sex_' + sex[0] : 1,
        'ChestPainType_' + chest_pain_type : 1,
        'RestingECG_' + resting_ecg : 1,
        'ExerciseAngina_' + exercise_angina[0] : 1,
        'ST_Slope_' + st_slope : 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[features] # this line ensures the order of columns matches the training data

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("The model predicts that the patient has heart disease.")
    else:
        st.success("The model predicts that the patient does not have heart disease.")


