# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1 :
Import the standard libraries such as pandas module to read the corresponding csv file.

Step 2 :
Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

Step 3 :
Import LabelEncoder and encode the corresponding dataset values.

Step 4 :
Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.

Step 5 :
Predict the values of array using the variable y_pred.

Step 6 :
Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.

Step 7 :
Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

Step 8:
End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:   ARHAM S
RegisterNumber:  212222110005
*/
```
```
import pandas as pd

data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:

HEAD OF THE DATA :

![264721805-8a1acdcf-6208-4ccd-a385-dce79acffb23](https://github.com/Adhithya4116/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707079/c2a52124-f0f7-4cca-9372-5537d2dd5d48)

COPY HEAD OF THE DATA:

![264722470-a8fbb70d-d081-404a-9852-d8e0cb514d72](https://github.com/Adhithya4116/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707079/da263507-864b-4ab0-88c0-e7b2d178ee0b)

NULL AND SUM :

![264722822-3dc72aad-5424-4b66-af78-4cff6e2b97a6](https://github.com/Adhithya4116/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707079/4f680650-2faa-424e-817a-2b0f172a2134)

DUPLICATED :

![264722958-c3040d19-87ba-41b2-9602-08be1862e318](https://github.com/Adhithya4116/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707079/fba59338-a0ff-4201-8f36-cf1924f7e610)

X VALUE:

![264725807-65c146b0-3c3d-4e28-9175-575ef0a3e1e9](https://github.com/Adhithya4116/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707079/b49cb2c9-27a0-45d8-8c5d-b9d22c9b2905)

Y VALUE :

![264725967-60c983e7-7e30-4915-a582-580814707244](https://github.com/Adhithya4116/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707079/9c08dfd4-23c0-44ab-a48e-45a23136586d)

PREDICTED VALUES :

![264726138-cb2539b7-5d40-4738-aaa6-a59f2d0bc329](https://github.com/Adhithya4116/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707079/aaeb9ad8-b861-43da-aebb-6aee4cd1f86f)

ACCURACY :

![264726318-ebcedcb0-3310-42ca-8122-9b3c9730b289](https://github.com/Adhithya4116/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707079/2b2578e5-9d1a-4f6f-925b-90e9e4b6805a)

CONFUSION MATRIX :

![264726756-deeaec70-cb55-4c5f-b019-bd1a8379d9ce](https://github.com/Adhithya4116/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707079/c05ef1a4-6107-4ec4-8d98-46fa2e6e3650)

**CLASSIFICATION REPORT :**

![264726957-18a5dede-f1c9-4211-bed3-43d7c14bef9c](https://github.com/Adhithya4116/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707079/f1325d9d-4d23-4a4c-951d-d1f0b56cb15b)

Prediction of LR :

![271896285-2bcc7f7f-26f7-46a9-ad12-2f6c01497de9](https://github.com/Adhithya4116/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707079/39d0ba5d-24b0-4836-89f9-e7c492403602)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
