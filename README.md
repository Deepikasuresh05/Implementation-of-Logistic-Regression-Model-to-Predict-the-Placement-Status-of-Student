# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).

Step 3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.

Step 4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.

Step 5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

Step 6. Stop

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Alan Samuel Vedanayagam
RegisterNumber: 212223040012
*/
```
```
import pandas as pd
df=pd.read_csv('Placement_Data.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/86c21228-880a-4831-8261-2c286f21dae8)
```
d1=df.copy()
d1=d1.drop(["sl_no","salary"],axis=1)
d1.head()
```
![image](https://github.com/user-attachments/assets/2f7cc339-56f4-4c16-a97b-3cff5c0a5c40)
```
d1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/182ba308-254b-411d-a2e5-96a9417d1a00)
```
d1.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/db0f4a32-519d-4179-a73b-1fdcc37c26c6)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
d1['gender']=le.fit_transform(d1["gender"])
d1["ssc_b"]=le.fit_transform(d1["ssc_b"])
d1["hsc_b"]=le.fit_transform(d1["hsc_b"])
d1["hsc_s"]=le.fit_transform(d1["hsc_s"])
d1["degree_t"]=le.fit_transform(d1["degree_t"])
d1["workex"]=le.fit_transform(d1["workex"])
d1["specialisation"]=le.fit_transform(d1["specialisation"])
d1["status"]=le.fit_transform(d1["status"])
d1
```
![image](https://github.com/user-attachments/assets/6c2d4250-b00a-4a90-b1d9-2a2abe3b9c9a)
```
x=d1.iloc[:, : -1]
x
```
![image](https://github.com/user-attachments/assets/cbe58a52-f336-404b-8ada-8c1848e8546b)
```
y=d1["status"]
y
```
![image](https://github.com/user-attachments/assets/5320629c-be6b-49c5-bf1c-afa8bfe7bfe6)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/fd161b01-6b2f-45f6-ae2d-d874f20bb1d3)
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/d6cb5529-9330-473c-a7ec-cd1b504257d1)
```
confusion=confusion_matrix(y_test,y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/51ca731e-9be7-4a74-9300-fa2d65bf6fc0)
```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
![image](https://github.com/user-attachments/assets/1bb3e4fd-4e1f-4f62-9ea1-bc1bb26f392c)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
