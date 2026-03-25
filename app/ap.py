import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("model.pkl")

st.title("Medical Insurance Cost Predictor")

# inputs
age = st.slider("Age", 18, 65, 30)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Children", 0, 5, 0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# prediction button
if st.button("Predict"):
    sample = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0
    }
    FEATURES = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']
    df = pd.DataFrame([sample])
    df = df[FEATURES]   

    prediction = model.predict(df)
    result = prediction[0]
    st.success(f"Estimated Insurance Cost: ₹{result:.2f}")
