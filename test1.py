import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Product Purchase Prediction')

model = joblib.load('regressor.pkl')
scalar = joblib.load('scaler.pkl')

age = st.number_input("Please Enter Age : ")
salary = st.number_input("Please Enter Salary : ")

btn = st.button("Predict")


if btn == True :
    datapoint_scaled = scalar.transform([[age,salary]])

    prediction = model.predict(datapoint_scaled)
    if round(prediction[0])==1 :
        st.write("Your Prediction : Purchased")
    else : 
        st.write("Your Prediction : Not Purchased")

