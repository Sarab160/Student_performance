from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# ---------- LOAD DATA ----------
df = pd.read_csv("student1.csv")

num_cols = [
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores",
    "Final_Exam_Score"
]

cat_cols = [
    "Gender",
    "Parental_Education_Level",
    "Internet_Access_at_Home",
    "Extracurricular_Activities"
]

X_num = df[num_cols]
X_cat = df[cat_cols]

ohe = OneHotEncoder(sparse_output=False, drop="first")
X_cat_enc = ohe.fit_transform(X_cat)
enc_cols = ohe.get_feature_names_out(cat_cols)

X_final = pd.concat(
    [X_num, pd.DataFrame(X_cat_enc, columns=enc_cols)],
    axis=1
)

le = LabelEncoder()
y = le.fit_transform(df["Pass_Fail"])

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

# ---------- ROUTE ----------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        name = request.form.get("name")

        input_data = {
            "Study_Hours_per_Week": float(request.form.get("study_hours")),
            "Attendance_Rate": float(request.form.get("attendance")),
            "Past_Exam_Scores": float(request.form.get("past_score")),
            "Final_Exam_Score": float(request.form.get("final_score")),
            "Gender": request.form.get("gender"),
            "Parental_Education_Level": request.form.get("parent_edu"),
            "Internet_Access_at_Home": request.form.get("internet"),
            "Extracurricular_Activities": request.form.get("extra")
        }

        input_df = pd.DataFrame([input_data])

        num_part = input_df[num_cols]
        cat_part = ohe.transform(input_df[cat_cols])
        cat_df = pd.DataFrame(cat_part, columns=enc_cols)

        final_input = pd.concat([num_part, cat_df], axis=1)

        result = model.predict(final_input)[0]
        result_label = le.inverse_transform([result])[0]

        prediction = f"{name}, based on your performance you might {result_label.upper()} the exam."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
