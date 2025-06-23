import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("house_price_model.pkl")

# App title
st.title("🏠 House Price Prediction App")
st.write("Enter the details below to estimate house price.")

# User input
sqft = st.number_input("Living Area (in sqft)", min_value=100, value=1500)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3, step=1)
bathrooms = st.number_input("Number of Full Bathrooms", min_value=0, value=2, step=1)

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame([[sqft, bedrooms, bathrooms]], columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
    prediction = model.predict(input_df)[0]
    st.success(f"🏷️ Estimated Price: ₹ {prediction:,.2f}")
