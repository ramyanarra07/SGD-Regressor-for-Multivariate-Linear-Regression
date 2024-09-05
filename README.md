# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1. Start

step 2. Data Preparation 

step 3. Hypothesis Definition

step 4. Cost Function

step 5. Parameter Update Rule

step 6. Iterative Training

step 7. Model Evaluation

step 8. End

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number
of occupants in the house with SGD regressor.
Developed by: Narra Ramya
RegisterNumber:212223040128


import pandas as pd
data=pd.read_csv("C:/Users/Admin/Desktop/Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])   
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
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
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
## y_pred
![image](https://github.com/user-attachments/assets/a830adf3-e86d-4bd2-a342-090e6083f98f)

## print(classification_report1)
![image](https://github.com/user-attachments/assets/f6ca2add-680f-4acd-bf21-bb11038aa904)

## lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
![image](https://github.com/user-attachments/assets/a2e7aff6-b78d-430d-b14c-521c06869cde)




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
