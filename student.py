import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix

df=pd.read_csv("student1.csv")

print(df.head())
print(df.info())

# sns.pairplot(data=df,hue="Pass_Fail")
# plt.show()

x=df[["Study_Hours_per_Week","Attendance_Rate","Past_Exam_Scores","Final_Exam_Score"]]
y=df["Pass_Fail"]

ohe=OneHotEncoder(sparse_output=False,drop="first")
fe=df[["Gender","Parental_Education_Level","Internet_Access_at_Home","Extracurricular_Activities"]]
encode_array=ohe.fit_transform(fe)
col=ohe.get_feature_names_out(fe.columns)
encode_data=pd.DataFrame(encode_array,columns=col)

X_final=pd.concat([x,encode_data],axis=1)

le=LabelEncoder()
y=le.fit_transform(y)
print(y)

x_train,x_test,y_train,y_test=train_test_split(X_final,y,random_state=42,test_size=0.2)

knc=KNeighborsClassifier(n_neighbors=1)
knc.fit(x_train,y_train)

print("Test Score",knc.score(x_test,y_test))
print("Train Score",knc.score(x_train,y_train))
print("Precision Score",precision_score(y_test,knc.predict(x_test)))
print("Recall Score",recall_score(y_test,knc.predict(x_test)))
print("F1 Score",f1_score(y_test,knc.predict(x_test)))
