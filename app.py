import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.markdown("### Predict whether a person is diabetic based on medical inputs")

st.divider()

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

st.divider()

if st.button("🔍 Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    prediction = model.predict(data)
    prob = model.predict_proba(data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.info(f"Confidence Score: {round(np.max(prob)*100, 2)}%")

st.divider()

st.caption("⚠️ This is a machine learning model for educational purposes only.")