import streamlit as st 
st.markdown(""" 
### Machine Learning Application  

This application predicts house prices using a 
trained regression model.""")

name = st.text_input("Enter your name")

st.write("Hello", name)
