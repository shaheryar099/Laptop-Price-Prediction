import pandas 
import numpy as np
from sklearn import * 
import streamlit as st
import pickle

df = pickle.load(open('df.pkl','rb'))
model = pickle.load(open('rf.pkl','rb'))


# ['Company', 'TypeName', 'Ram', 'Weight', 'Price', 'Touchscreen', 'Ips',
 #      'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']

st.title('Laptop Price Prediction')
st.header('Fill details to predict price of laptop')
Company = st.selectbox('Company',df['Company'].unique())
TypeName = st.selectbox('TypeName',df['TypeName'].unique())
Ram = st.selectbox('Ram(in GB)',[8, 16,  4,  2, 12,  6, 32, 24, 64])
Weight = st.number_input('Weight of the laptop')
Touchscreen = st.selectbox('Touchscreen',['Yes','No'])
Ips = st.selectbox('Ips',['Yes','No'])
Cpu  = st.selectbox('CPU brand',df['Cpu brand'].unique())
hdd = st.selectbox('HDD',[0, 32, 128, 500, 1000, 2000])
ssd = st.selectbox('SSD',[0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1000, 1024])
Gpu = st.selectbox('GPU brand',df['Gpu brand'].unique())
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Laptop Price'):
    
    if Touchscreen == 'Yes':
        Touchscreen=1
    else:
        Touchscreen=0
    if Ips == 'Yes':
        Ips=1
    else:
        Ips=0
    test_data = np.array([Company, TypeName, Ram, Weight,Touchscreen, Ips, Cpu, hdd, ssd, Gpu, os])
    test_data = test_data.reshape([1,11])
    
    
    st.success(model.predict(test_data)[0])
    
