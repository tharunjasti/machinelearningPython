# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing libiraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#imorting dataset
df=pd.read_csv('Salary_Data.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,1].values

'''
# handling missing data 
# Remove data or replace with mean 
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:, 0]  =labelencoder_X.fit_transform(X[:,0])
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
'''

# Spliting the dataset into Training and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
'''

# Fitting Simple Linear Regresion to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

# Predecting the Test set results
y_pred=regressor.predict(X_test)

# Visualising the Traing set results
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs experence (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Years of experience')
plt.show()

# Visualising theTest set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs experence (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Years of experience')
plt.show()