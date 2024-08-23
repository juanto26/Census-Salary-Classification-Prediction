import streamlit as st
import eda
import predict

navigation = st.sidebar.selectbox('select page:', ('EDA','Predict'))

if navigation == 'EDA':
    eda.run()
else :
    predict.run()