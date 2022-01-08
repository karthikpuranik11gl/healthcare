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
def predict_note_authentication(rooms,department,ward,doctor,staff,id,age,gender,type,severity,conditions,visitors,insurance,deposit):
    l=[rooms,department,ward,doctor,staff,id,age,gender,type,severity,conditions,visitors,insurance,deposit]
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
    rooms = st.number_input("Available Extra Rooms in Hospital")
    department = st.selectbox("Department",("gynecology", "radiotherapy", "anesthesia","TB & Chest disease", "surgery" ))
    ward = st.selectbox("Ward Facility Code",("A","B","C","D", "E", "F"))
    doctor = st.selectbox("Doctor Name",("Dr Sarah", "Dr Olivia", "Dr Sophia", "Dr Nathan" ,"Dr Sam", "Dr John", "Dr Mark", "Dr Isaac", "Dr Simon"))
    staff = st.number_input("Staff available")
    id = st.number_input("Patient ID")
    age = st.selectbox("Age",("0-10","11-20","21-30","31-40","41-50","51-60","61-70", "71-80", "81-90","91-100"))
    gender=st.selectbox("Gender", ("Male","Female", "Other"))
    type=st.selectbox("Type of admission", ("Trauma", "Emergency", "Urgent"))
    severity=st.selectbox("Severty of illness", ("Minor", "Moderate", "Extreme"))
    conditions=st.selectbox("Previous health conditions", ("High Blood Pressure","Diabetes","Asthama","Heart disease", "None"," Other"))
    visitors=st.number_input("Number of visitors")
    insurance=st.selectbox("Health insurance?", ("Yes", "No"))
    deposit=st.number_input("Admission deposit")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(rooms,department,ward,doctor,staff,id,age,gender,type,severity,conditions,visitors,insurance,deposit)
    st.success('The output is {}'.format(result)+ " days")
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    