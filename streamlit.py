import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Student Pass/Fail Prediction", layout="centered")
st.title("🎓 Student Pass/Fail Prediction (KNN)")

# --------------------------------------------------
# LOAD DATA (NO BROWSE)
# --------------------------------------------------
df = pd.read_csv("student1.csv")

# --------------------------------------------------
# FEATURES & TARGET
# --------------------------------------------------
x = df[
    [
        "Study_Hours_per_Week",
        "Attendance_Rate",
        "Past_Exam_Scores",
        "Final_Exam_Score"
    ]
]

y = df["Pass_Fail"]

categorical_features = [
    "Gender",
    "Parental_Education_Level",
    "Internet_Access_at_Home",
    "Extracurricular_Activities"
]

# --------------------------------------------------
# ENCODING
# --------------------------------------------------
ohe = OneHotEncoder(sparse_output=False, drop="first")
encoded_array = ohe.fit_transform(df[categorical_features])
encoded_cols = ohe.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

X_final = pd.concat([x.reset_index(drop=True), encoded_df], axis=1)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42
)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

# --------------------------------------------------
# MODEL METRICS
# --------------------------------------------------
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Train Accuracy", f"{knn.score(x_train, y_train):.2f}")
col2.metric("Test Accuracy", f"{knn.score(x_test, y_test):.2f}")
col3.metric("Precision", f"{precision_score(y_test, knn.predict(x_test)):.2f}")
col4.metric("Recall", f"{recall_score(y_test, knn.predict(x_test)):.2f}")

st.metric("F1 Score", f"{f1_score(y_test, knn.predict(x_test)):.2f}")

# --------------------------------------------------
# USER INPUT SECTION
# --------------------------------------------------
st.subheader("🧑‍🎓 Enter Student Details")

study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=80.0)
attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0)
past_score = st.number_input("Past Exam Score", min_value=0.0, max_value=100.0)
final_score = st.number_input("Final Exam Score", min_value=0.0, max_value=100.0)

gender = st.selectbox("Gender", df["Gender"].unique())
parent_edu = st.selectbox(
    "Parental Education Level",
    df["Parental_Education_Level"].unique()
)
internet = st.selectbox(
    "Internet Access at Home",
    df["Internet_Access_at_Home"].unique()
)
extra = st.selectbox(
    "Extracurricular Activities",
    df["Extracurricular_Activities"].unique()
)

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------
if st.button("🔮 Predict Pass / Fail"):
    input_df = pd.DataFrame(
        [[
            study_hours,
            attendance,
            past_score,
            final_score,
            gender,
            parent_edu,
            internet,
            extra
        ]],
        columns=[
            "Study_Hours_per_Week",
            "Attendance_Rate",
            "Past_Exam_Scores",
            "Final_Exam_Score",
            "Gender",
            "Parental_Education_Level",
            "Internet_Access_at_Home",
            "Extracurricular_Activities"
        ]
    )

    input_encoded = ohe.transform(input_df[categorical_features])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_cols)

    input_final = pd.concat(
        [input_df[x.columns], input_encoded_df],
        axis=1
    )

    prediction = knn.predict(input_final)
    result = le.inverse_transform(prediction)[0]

    st.success(f"✅ Prediction Result: **{result}**")
