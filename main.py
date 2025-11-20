import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn import *

# Load preprocessed data and model
df = pickle.load(open('df.pkl', 'rb'))
model = pickle.load(open('rf.pkl', 'rb'))

st.title('💻 Laptop Price Prediction')
st.header('Fill the details to predict the price of the laptop')

company = st.selectbox('Company Name', df['Company'].sort_values().unique())
type_ = st.selectbox('Type Name', df['TypeName'].unique())
ram = st.selectbox('RAM (GB)', df['Ram'].sort_values().unique())
weight = st.number_input('Enter the weight (KG)')
touchscreen = st.selectbox('Touch Screen', ['Yes', 'No'])
ips = st.selectbox('IPS', ['Yes', 'No'])
cpubrand = st.selectbox('CPU Brand', df['Cpu brand'].sort_values().unique())
hdd = st.selectbox('HDD (GB)', [0, 32, 128, 500, 1000, 2000])
ssd = st.selectbox('SSD (GB)', [0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1000, 1024])
gpu = st.selectbox('GPU Brand', df['Gpu brand'].sort_values().unique())
os = st.selectbox('OS', df['os'].sort_values().unique())

if st.button('Predict the Laptop Price'):
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # ⚠️ Here you need the same preprocessing as training
    test_data = np.array([company, type_, ram, weight, touchscreen, ips, cpubrand, hdd, ssd, gpu, os]).reshape(1, -1)

    prediction = model.predict(test_data)[0]
    st.success(f'Predicted Laptop Price: ₹{int(prediction)}')
