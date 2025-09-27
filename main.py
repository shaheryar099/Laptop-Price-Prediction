import pandas as pd
import numpy as np
from sklearn import *
import streamlit as st
import pickle


df = pickle.load(open('df.pkl','rb'))
model = pickle.load(open('rf.pkl','rb'))

# Index(['Company', 'TypeName', 'Ram', 'Weight', 'Price', 'Touchscreen', 'Ips' 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'], dtype='object')

st.title('Laptop Price Prediction')
st.header('Fill the details to predict the price of the laptop')
company = st.selectbox('Company Name',df['Company'].sort_values().unique())
type = st.selectbox('Name Type',df['TypeName'].unique())
ram = st.selectbox('Ram',df['Ram'].sort_values().unique())
weight = st.number_input('Enter the weight (KG)')
touchscreen = st.selectbox('Touch Screen',['Yes', 'No'])  #Yes/No
ips = st.selectbox('IPS',['Yes', 'No']) #Yes/No
cpubrand = st.selectbox('CPU Brand',df['Cpu brand'].sort_values().unique())
hdd = st.selectbox('HDD',[0, 32, 128, 500, 1000, 2000])
ssd = st.selectbox('SSD',[0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1000, 1024])
gpu = st.selectbox('GPU Brand', df['Gpu brand'].sort_values().unique())
os = st. selectbox('OS',df['os'].sort_values().unique())

if st.button('Predict the laptop Price'):

    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    test_data = np.array([company, type, ram, weight, touchscreen, ips, cpubrand, hdd, ssd, gpu, os])
    test_data = test_data.reshape(1, -1)

    prediction = model.predict(test_data)[0]
    st.success(f'Predicted Laptop Price: ₹{int(prediction)}')

