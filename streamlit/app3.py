import streamlit as st
from datetime import date
st.title("Student Registration System")
st.header("Student Details")
st.markdown("### Enter the following information")
st.caption("This application demonstrates basic Streamlit widgets.")
name = st.text_input("Student Name")
age = st.number_input(
    "Age",
    min_value=15,
    max_value=60,
    value=20
)
cgpa = st.slider(
    "CGPA",
    0.0,
    10.0,
    7.5
)
gender = st.radio(
    "Gender",
    ["Male", "Female", "Other"]
)
dob = st.date_input(
    "Date of Birth",
    date(2000,1,1)
)
agree = st.checkbox("I confirm the information is correct.")
st.subheader("Python Example")
st.code("""
print("Welcome to Streamlit")
""", language="python")
st.text("Click Submit to save details.")
if st.button("Submit"):
    if name == "":
        st.error("Please enter your name.")
    elif not agree:
        st.warning("Please accept the declaration.")
    else:
        st.success("Registration Successful")
        st.info("Student Information")
        st.write({
            "Name": name,
            "Age": age,
            "CGPA": cgpa,
            "Gender": gender,
            "DOB": dob
        })