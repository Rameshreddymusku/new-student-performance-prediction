import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("../models/best_model.pkl")

# Feature list (must match EXACT training order)
feature_names = [
    'Hours_Studied',
    'Attendance',
    'Parental_Involvement',
    'Access_to_Resources',
    'Extracurricular_Activities',
    'Sleep_Hours',
    'Previous_Scores',
    'Motivation_Level',
    'Internet_Access',
    'Tutoring_Sessions',
    'Family_Income',
    'Teacher_Quality',
    'School_Type',
    'Peer_Influence',
    'Physical_Activity',
    'Learning_Disabilities',
    'Parental_Education_Level',
    'Distance_from_Home',
    'Gender'
]

# LabelEncoder helper (mimics your training loop)
def encode_label(value, possible_values):
    le = LabelEncoder()
    le.fit(possible_values)
    return int(le.transform([value])[0])

st.title("🎓 Student Exam Score Predictor")

# -----------------------
# USER INPUTS
# -----------------------

Hours_Studied = st.number_input("Hours Studied", min_value=0, max_value=24)
Attendance = st.number_input("Attendance (%)", min_value=0, max_value=100)

Parental_Involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
Access_to_Resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
Extracurricular_Activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])

Sleep_Hours = st.number_input("Sleep Hours", min_value=0, max_value=24)
Previous_Scores = st.number_input("Previous Scores", min_value=0, max_value=100)
Motivation_Level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])

Internet_Access = st.selectbox("Internet Access", ["Yes", "No"])
Tutoring_Sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=20)
Family_Income = st.number_input("Family Income", min_value=0)

Teacher_Quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
School_Type = st.selectbox("School Type", ["Public", "Private"])
Peer_Influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
Physical_Activity = st.selectbox("Physical Activity", ["Yes", "No"])
Learning_Disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
Parental_Education_Level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
Distance_from_Home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
Gender = st.selectbox("Gender", ["Male", "Female"])

# -----------------------
# ENCODE VALUES
# -----------------------

encoded_row = {
    'Hours_Studied': Hours_Studied,
    'Attendance': Attendance,
    'Parental_Involvement': encode_label(Parental_Involvement, ["High", "Low", "Medium"]),
    'Access_to_Resources': encode_label(Access_to_Resources, ["High", "Low", "Medium"]),
    'Extracurricular_Activities': encode_label(Extracurricular_Activities, ["No", "Yes"]),
    'Sleep_Hours': Sleep_Hours,
    'Previous_Scores': Previous_Scores,
    'Motivation_Level': encode_label(Motivation_Level, ["High", "Low", "Medium"]),
    'Internet_Access': encode_label(Internet_Access, ["No", "Yes"]),
    'Tutoring_Sessions': Tutoring_Sessions,
    'Family_Income': Family_Income,
    'Teacher_Quality': encode_label(Teacher_Quality, ["High", "Low", "Medium"]),
    'School_Type': encode_label(School_Type, ["Private", "Public"]),
    'Peer_Influence': encode_label(Peer_Influence, ["Negative", "Neutral", "Positive"]),
    'Physical_Activity': encode_label(Physical_Activity, ["No", "Yes"]),
    'Learning_Disabilities': encode_label(Learning_Disabilities, ["No", "Yes"]),
    'Parental_Education_Level': encode_label(Parental_Education_Level, ["College", "High School", "Postgraduate"]),
    'Distance_from_Home': encode_label(Distance_from_Home, ["Far", "Moderate", "Near"]),
    'Gender': encode_label(Gender, ["Female", "Male"])
}

# Convert to DataFrame and Reorder Columns
input_df = pd.DataFrame([encoded_row])[feature_names]

# -----------------------
# PREDICT
# -----------------------

if st.button("Predict Score"):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"📘 Predicted Exam Score: **{round(pred, 2)}**")
    except Exception as e:
        st.error(f"Error: {e}")