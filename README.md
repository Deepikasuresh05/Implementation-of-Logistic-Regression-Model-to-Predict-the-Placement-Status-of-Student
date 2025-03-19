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

### Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
#### Developed by: S Deepika
#### RegisterNumber: 212223230039

```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
## OUTPUT:


# DATA HEAD
![image](https://github.com/user-attachments/assets/f78e5f92-9dd3-4f39-8dae-a01edbfb1e08)

# DATA1 HEAD
![image](https://github.com/user-attachments/assets/3920fd1a-44d4-4c79-8296-8212ae774093)

# ISNULL().SUM()
![image](https://github.com/user-attachments/assets/b5b504eb-15d8-4274-b718-92f1d6b275b6)

# DATA DUPLICATE

![image](https://github.com/user-attachments/assets/cb6f4829-ee11-4252-8fa8-127cbd87697e)

# PRINT DATA

![image](https://github.com/user-attachments/assets/f4d8def8-3789-4f61-9ef8-6fb060718f6b)

# STATUS

![image](https://github.com/user-attachments/assets/f22c4c5c-1e60-437b-a4c7-02e198c73110)

# Y_PRED

![image](https://github.com/user-attachments/assets/9d396877-4d02-4ed4-92ea-dbbe7534ee1a)

# ACCURACY

![image](https://github.com/user-attachments/assets/bdfa007e-3ee3-4d82-8b4d-378b9ae3d0d5)

# CONFUSION MATRIX

![image](https://github.com/user-attachments/assets/6da6ad6a-66fa-43f7-a6d4-56363113f979)

# CLASSIFICATION

![image](https://github.com/user-attachments/assets/b1398038-74d4-415b-8d4d-700ceb1bb86f)

# LR PREDICT

![image](https://github.com/user-attachments/assets/610db60b-96c3-4c75-8694-7e7394d3e188)















## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
