from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# ================= LOAD & TRAIN MODEL =================
df = pd.read_csv("student1.csv")

x = df[["Study_Hours_per_Week", "Attendance_Rate", "Past_Exam_Scores", "Final_Exam_Score"]]
y = df["Pass_Fail"]

fe = df[["Gender", "Parental_Education_Level", "Internet_Access_at_Home", "Extracurricular_Activities"]]

ohe = OneHotEncoder(sparse_output=False, drop="first")
encoded_array = ohe.fit_transform(fe)
encoded_cols = ohe.get_feature_names_out(fe.columns)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

X_final = pd.concat([x, encoded_df], axis=1)

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)

# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    name = ""

    if request.method == "POST":
        name = request.form["name"]

        input_data = {
            "Study_Hours_per_Week": float(request.form["study_hours"]),
            "Attendance_Rate": float(request.form["attendance"]),
            "Past_Exam_Scores": float(request.form["past_score"]),
            "Final_Exam_Score": float(request.form["final_score"]),
            "Gender": request.form["gender"],
            "Parental_Education_Level": request.form["parent_edu"],
            "Internet_Access_at_Home": request.form["internet"],
            "Extracurricular_Activities": request.form["extra"]
        }

        input_df = pd.DataFrame([input_data])

        encoded_input = ohe.transform(
            input_df[["Gender", "Parental_Education_Level",
                      "Internet_Access_at_Home", "Extracurricular_Activities"]]
        )

        encoded_input_df = pd.DataFrame(
            encoded_input,
            columns=ohe.get_feature_names_out()
        )

        final_input = pd.concat([
            input_df[["Study_Hours_per_Week", "Attendance_Rate",
                      "Past_Exam_Scores", "Final_Exam_Score"]],
            encoded_input_df
        ], axis=1)

        result = model.predict(final_input)[0]
        label = le.inverse_transform([result])[0]

        if label.lower() == "fail":
            prediction = f"{name}, you might fail in upcoming exams due to your performance."
        else:
            prediction = f"{name}, you are likely to pass. Keep up the good work!"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
