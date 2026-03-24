import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("../model/model.pkl")

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
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }

    df = pd.DataFrame([sample])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(df)[0]

    st.success(f"Estimated Insurance Cost: {prediction:.2f}")
