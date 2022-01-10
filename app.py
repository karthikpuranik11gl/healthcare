import numpy as np
import pandas as pd
import streamlit as st 
import joblib

from PIL import Image

regressor=joblib.load("regressor.pkl")
x_train=joblib.load("dataset.pkl")


def welcome():
    return "Welcome All"

@st.cache
def predict_note_authentication(rooms,department,ward,doctor,staff,idd,age,gender,typee,severity,conditions,visitors,insurance,deposit):
    l=[rooms,department,ward,doctor,staff,idd,age,gender,typee,severity,conditions,visitors,insurance,deposit]
    cols=['Available Extra Rooms in Hospital', 'Department', 'Ward_Facility_Code', 'doctor_name', 'staff_available', 'patientid', 'Age', 'gender', 'Type of Admission', 'Severity of Illness', 'health_conditions',
       'Visitors with Patient', 'Insurance', 'Admission_Deposit']
    data2=pd.DataFrame([l], columns=cols)
    data2 = pd.get_dummies( data2, columns=data2.select_dtypes(include=["object", "category"]).columns.tolist(), drop_first=True,)
    data2 = data2.reindex(columns = x_train.columns, fill_value=0)
    prediction=regressor.predict(data2)
    print(prediction)
    return int(round(prediction[0], 0))



def main():
    #st.title("Length of stay predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Length of stay predictor </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    rooms = st.number_input("Available Extra Rooms in Hospital", min_value=0, max_value=25)
    department = st.selectbox("Department",("gynecology", "radiotherapy", "anesthesia","TB & Chest disease", "surgery" ))
    ward = st.selectbox("Ward Facility Code",("A","B","C","D", "E", "F"))
    doctor = st.selectbox("Doctor Name",("Dr Sarah", "Dr Olivia", "Dr Sophia", "Dr Nathan" ,"Dr Sam", "Dr John", "Dr Mark", "Dr Isaac", "Dr Simon"))
    staff = st.number_input("Staff available", min_value=0, max_value=15)
    idd = st.number_input("Patient ID", min_value=30000, max_value=99999)
    age = st.selectbox("Age",("0-10","11-20","21-30","31-40","41-50","51-60","61-70", "71-80", "81-90","91-100"))
    gender=st.selectbox("Gender", ("Male","Female", "Other"))
    type=st.selectbox("Type of admission", ("Trauma", "Emergency", "Urgent"))
    severity=st.selectbox("Severty of illness", ("Minor", "Moderate", "Extreme"))
    conditions=st.selectbox("Previous health conditions", ("High Blood Pressure","Diabetes","Asthama","Heart disease", "None"," Other"))
    visitors=st.number_input("Number of visitors", min_value=0, max_value=50)
    insurance=st.selectbox("Health insurance?", ("Yes", "No"))
    deposit=st.number_input("Admission deposit", min_value=0, max_value=100000)
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(rooms,department,ward,doctor,staff,idd,age,gender,typee,severity,conditions,visitors,insurance,deposit)
    st.success('The output is {}'.format(result)+ " days")
    if st.button("About"):
        st.text("Data dictionary:")
        st.text("patientid: Patient ID")
        st.text("Age: Range of age of the patient")
        st.text("gender: Gender of the patient.")
        st.text("Type of Admission: Trauma, emergency or urgent.")
        st.text("Severity of Illness: Extreme, moderate or minor.")
        st.text("health_conditions: Any previous health conditions suffered by the patient.")
        st.text("Visitors with Patient: The number of patients who accompany the patient.")
        st.text("Insurance: Does the patient have health insurance or not?")
        st.text("Admission_Deposit: The deposit paid by the patient during admission.")
        st.text("Available Extra Rooms in Hospital: The number of rooms available during admission.")
        st.text("Department: The department which will be treating the patient.")
        st.text("Ward_Facility_Code: The code of the ward facility in which the patient will be admitted.")
        st.text("doctor_name: The doctor who will be treating the patient.")
        st.text("staff_available: The number of staff who are not occupied at the moment in the ward.")
        st.text("Stay (in days): The number of days that the patient has stayed in the hospital.")
        

if __name__=='__main__':
    main()
    
    
    
