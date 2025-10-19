import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# -------------------------------
# Load prebuilt ANN model and scaler
# -------------------------------
model = tf.keras.models.load_model('regression_model.h5')

with open('label_encoded_gender.pkl', 'rb') as file:
    label_encoded_gender = pickle.load(file)

with open('onehot_encoded_geo.pkl', 'rb') as file:
    onehot_encoded_geo = pickle.load(file)

with open('Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("💰 Customer Estimated Salary Prediction")
st.write("Predict the estimated salary of a customer using ANN based on Churn Modelling dataset.")

# -------------------------------
# User Inputs (features from Churn Modelling.csv)
# -------------------------------
geography = st.selectbox('Geography', onehot_encoded_geo.categories_[0])
gender = st.selectbox('Gender', label_encoded_gender.classes_)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, step=1)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure (Years with Bank)', 0, 10)
balance = st.number_input('Account Balance', min_value=0.0, step=100.0)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# -------------------------------
# Preprocess Input
# -------------------------------
# Encode gender
gender_encoded = label_encoded_gender.transform([gender])[0]

# Create DataFrame (without Geography)
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

# One-hot encode Geography
geo_encoded = onehot_encoded_geo.transform([[geography]])
if hasattr(geo_encoded, "toarray"):
    geo_encoded = geo_encoded.toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoded_geo.get_feature_names_out(['Geography'])
)

# Combine all features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Align columns to scaler expectation
input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_data)

# -------------------------------
# Predict Salary
# -------------------------------
predicted_salary = model.predict(input_scaled)
predicted_salary_value = predicted_salary[0][0]

# -------------------------------
# Display Output
# -------------------------------
st.subheader("🔍 Predicted Estimated Salary")
st.write(f"💵 The estimated salary is: **${predicted_salary_value:,.2f}**")

st.caption("Model powered by TensorFlow & Streamlit | Dataset: Churn Modelling.csv")
