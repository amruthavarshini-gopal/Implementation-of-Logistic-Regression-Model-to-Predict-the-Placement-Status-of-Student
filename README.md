# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## Aim:

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Load the Dataset

2.Create a Copy of the Original Data

3.Drop Irrelevant Columns (sl_no, salary)

4.Check for Missing Values

5.Check for Duplicate Rows

6.Encode Categorical Features using Label Encoding

7.Split Data into Features (X) and Target (y)

8.Split Data into Training and Testing Sets

9.Initialize and Train Logistic Regression Model

10.Make Predictions on Test Set

11.Evaluate Model using Accuracy Score

12.Generate and Display Confusion Matrix

13.Generate and Display Classification Report

14.Make Prediction on a New Sample Input

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Amruthavarshini Gopal
RegisterNumber: 212223230013  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Placement_Data.csv")
df.head()
df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()
df1.isnull().sum()
df1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1
x=df1.iloc[:,:-1]
x
y=df1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
confusion=confusion_matrix(y_test,y_pred)
confusion
lr.predict([[1,80,1,9,1,1,90,1,0,85,1,85]])
```

## Output:

### Placement Dataset

![image](https://github.com/user-attachments/assets/a5ce4eb8-0686-4060-bf71-e7362d0c6a8d)


### Dataset after Feature Selection

![Screenshot 2025-04-04 092403](https://github.com/user-attachments/assets/1ae6f254-d6a3-46f9-9fd8-243e3dd2a9d9)

### Null count

![Screenshot 2025-04-04 092434](https://github.com/user-attachments/assets/7276ee54-4e00-48ff-964d-fa49af67b8f5)

### Duplicated count

![Screenshot 2025-04-04 092504](https://github.com/user-attachments/assets/a1f80e03-c885-458b-9d33-81b3014e58c7)

### Dataset after Label Encoding

![Screenshot 2025-04-04 092525](https://github.com/user-attachments/assets/add98a21-2853-4df7-b0cb-9da29b696238)

### X Data

![Screenshot 2025-04-04 092721](https://github.com/user-attachments/assets/76df3974-2444-440f-8d8c-14b4123cf2e8)

### Y Data

![Screenshot 2025-04-04 092849](https://github.com/user-attachments/assets/8dedac5c-814b-4ba9-be76-d886b0ae78a0)

### Y Predicted

![Screenshot 2025-04-04 092907](https://github.com/user-attachments/assets/d13ac391-4752-4173-8c68-09d77fafad72)

### Accuracy

![Screenshot 2025-04-04 092940](https://github.com/user-attachments/assets/09f14ac0-5df2-40c4-b22d-baa1a23337bd)

### Confusion Matrix

![Screenshot 2025-04-04 093036](https://github.com/user-attachments/assets/e0c46740-6415-427d-85b8-d388671b9643)

### Classification

![Screenshot 2025-04-04 093055](https://github.com/user-attachments/assets/685775f9-ddcc-4ddf-ba58-83936dfb41f3)

### Prediction of logistic regression

![image](https://github.com/user-attachments/assets/38561a04-29e1-4faa-8f42-506dfe67704c)


## Result:

Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
