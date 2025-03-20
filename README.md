# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SONU S
RegisterNumber: 212223220107

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())

x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


*/
```

## Output:
HEAD VALUES:
![Screenshot 2025-03-20 025557](https://github.com/user-attachments/assets/5cae96f1-617e-4c28-8d0a-fbdcdcda5459)
TAIL VALUES:
![Screenshot 2025-03-20 025605](https://github.com/user-attachments/assets/efccfd2d-ca19-4eb9-9252-4ca6edb2cf58)
HOURS VALUES:
![Screenshot 2025-03-20 025615](https://github.com/user-attachments/assets/21124d7b-e221-4498-af9e-cb9d9324ee18)
Y_PREDICTION:
![Screenshot 2025-03-20 025626](https://github.com/user-attachments/assets/8d9ee1c9-5d63-4081-8b98-76e4c80362e4)
RESULT OF MSE,MAE,RMSE:
![Screenshot 2025-03-20 025635](https://github.com/user-attachments/assets/9f69e33a-27bc-4cb3-ae19-6a18ab866a83)

![Screenshot 2025-03-20 025644](https://github.com/user-attachments/assets/6f3493cd-67cd-4bc8-bfff-5222e3b21652)
![Screenshot 2025-03-20 030915](https://github.com/user-attachments/assets/4eb70393-b531-4864-aa39-ebaeecca76f2)

![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
