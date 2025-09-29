import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load data and model
df = pickle.load(open('df.pkl','rb'))
model = pickle.load(open('rf.pkl','rb'))

st.title('💻 Laptop Price Prediction')
st.header('Fill details to predict price of laptop')

# User Inputs
Company = st.selectbox('Company', df['Company'].unique())
TypeName = st.selectbox('TypeName', df['TypeName'].unique())
Ram = st.selectbox('Ram (GB)', [2,4,6,8,12,16,24,32,64])
Weight = st.number_input('Weight of the laptop (kg)')
Touchscreen = st.selectbox('Touchscreen', ['Yes','No'])
Ips = st.selectbox('IPS Display', ['Yes','No'])
Cpu  = st.selectbox('CPU brand', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (GB)', [0,32,128,500,1000,2000])
ssd = st.selectbox('SSD (GB)', [0,8,16,32,64,128,180,240,256,512,768,1000,1024])
Gpu = st.selectbox('GPU brand', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

if st.button('Predict Laptop Price'):
    # Convert Yes/No to 0/1
    Touchscreen = 1 if Touchscreen == 'Yes' else 0
    Ips = 1 if Ips == 'Yes' else 0

    # Prepare input in correct format (must match model training preprocessing)
    test_data = pd.DataFrame([[Company, TypeName, Ram, Weight, Touchscreen, Ips, Cpu, hdd, ssd, Gpu, os]],
                             columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])
    
    # ⚠️ IMPORTANT: Apply the same encoding as training (not shown here)
    # Example: test_data = encoder.transform(test_data)
    
    prediction = model.predict(test_data)[0]
    st.success(f"💰 Predicted Laptop Price: {int(prediction)}")
