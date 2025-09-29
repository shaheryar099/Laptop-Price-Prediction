import pandas as pd
import numpy as np
import streamlit as st
import joblib   # better than pickle for sklearn objects

# -----------------------------
# Load dataset and trained model
# -----------------------------
df = joblib.load(open('df.pkl', 'rb'))     # dataset
model = joblib.load(open('rf.pkl', 'rb'))  # trained RandomForest model

# -----------------------------
# Streamlit UI
# -----------------------------
st.title('💻 Laptop Price Prediction')
st.header('Fill in details to predict the price of a laptop')

# User Inputs
Company = st.selectbox('Company', df['Company'].unique())
TypeName = st.selectbox('Type Name', df['TypeName'].unique())
Ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
Weight = st.number_input('Weight of the laptop (kg)', min_value=0.5, max_value=5.0, step=0.1)
Touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])
Ips = st.selectbox('IPS Display', ['Yes', 'No'])
Cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (GB)', [0, 32, 128, 500, 1000, 2000])
ssd = st.selectbox('SSD (GB)', [0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1000, 1024])
Gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

# -----------------------------
# Prediction
# -----------------------------
if st.button('Predict Laptop Price'):
    # Convert Yes/No to binary
    Touchscreen = 1 if Touchscreen == 'Yes' else 0
    Ips = 1 if Ips == 'Yes' else 0

    # Create dataframe (same format as training)
    test_data = pd.DataFrame([[Company, TypeName, Ram, Weight, Touchscreen, Ips, Cpu, hdd, ssd, Gpu, os]],
                             columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips',
                                      'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

    # ⚠️ IMPORTANT: Model must already include preprocessing (LabelEncoding/OneHotEncoding)
    prediction = model.predict(test_data)[0]

    st.success(f"💰 Predicted Laptop Price: {int(prediction):,} INR")
