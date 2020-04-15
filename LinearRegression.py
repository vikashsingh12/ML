# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:35:37 2020

@author: LENOVO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('employees.csv')

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection  import train_test_split
X_train,x_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=0) 

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X_train,y_train)

y_pred= lin_reg.predict(x_test)

#train data graph
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,lin_reg.predict(X_train), color='blue')
plt.title('Linear regression salary vs Expectation')
plt.xlabel("Years of Employee")
plt.ylabel("Saleries of employee")
plt.show()

#test data graph
plt.scatter(x_test,y_test,color='red')
plt.plot(X_train,lin_reg.predict(X_train), color='blue')
plt.title('Linear regression salary vs Expectation')
plt.xlabel("Years of Employee")
plt.ylabel("Saleries of employee")
plt.show()
