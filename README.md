# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Jhagan B
RegisterNumber:  212220040066
*/

import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

Data Head:

![image](https://github.com/jhaganb/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/63654882/416d7bd1-02ac-4230-beec-22ca77142658)

DataSet Info:

![image](https://github.com/jhaganb/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/63654882/12d8735f-d183-4332-88fe-abf31e2f6dcf)

Null dataset:

![image](https://github.com/jhaganb/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/63654882/450136d1-ef98-47bf-9f57-ca70150d45c7)

values count in left column:

![image](https://github.com/jhaganb/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/63654882/bd3c64d4-3665-4dc1-893c-92d6494afb46)

X.Head:

![image](https://github.com/jhaganb/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/63654882/3bf680aa-2021-420f-8da7-82573e06ba9f)

Dataset transformed head:

![image](https://github.com/jhaganb/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/63654882/8e691788-94a4-459d-b575-8378b5bab3a7)

Accuracy:

![image](https://github.com/jhaganb/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/63654882/97f82051-be49-481f-ac87-c3b40fa65ec0)

Data Prediction:

![image](https://github.com/jhaganb/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/63654882/623695ff-303d-4038-b929-61ee39ee94b5)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
