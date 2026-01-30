# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Implementation of Simple Linear Regression Model for Predicting the Marks Scored


Step 1

Start the program.

Step 2

Import the required libraries such as pandas, numpy, matplotlib, and sklearn.

Step 3

Load the student scores dataset.

Step 4

Display the dataset to understand its structure.

Step 5

Separate the independent variable (Hours) and dependent variable (Marks).

Step 6

Split the dataset into training set and testing set.

Step 7

Create the Simple Linear Regression model.

Step 8

Train the model using the training data.

Step 9

Predict the marks for the testing data.

Step 10

Display the result using graphs and error values, then stop the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SAIKRIPA SK
RegisterNumber:  212224040284
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\admin\ML\DATASET-20260129\student_scores.csv")



X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0
)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

plt.scatter(X_train, Y_train)
plt.plot(X_train, regressor.predict(X_train))
plt.title("Training Set")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test, Y_test)
plt.plot(X_test, regressor.predict(X_test))
plt.title("Testing Set")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("MAE:", mean_absolute_error(Y_test, Y_pred))
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_test, Y_pred)))

```


## Output:
<img width="941" height="580" alt="image" src="https://github.com/user-attachments/assets/1866fd2f-eee2-411a-bcdd-3cc1a530de2b" />
<img width="877" height="675" alt="image" src="https://github.com/user-attachments/assets/316d0898-01e9-4cb2-bbda-bd7ebc136fd1" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
