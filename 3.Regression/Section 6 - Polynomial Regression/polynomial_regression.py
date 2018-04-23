# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Position_Salaries.csv')

X=df[['Level']].values
y=df[['Salary']].values

# Fitting Linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#ploting for linear regression
plt.scatter(X,y, color='blue')
plt.plot(X,lin_reg.predict(X), color='red')
plt.title('Multi Linear Regression')
plt.xlabel('Yeras')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with linear regression
lin_reg.predict(6.5)



# Fitting Polynomnial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)


#ploting for polynomial feature regression
X_grid=np.arange(min(X),max(X), step=0.1)
X_grid=X_grid.reshape((len(X_grid),1)) #replace X with X_grid for plot
plt.scatter(X,y, color='blue')
plt.plot(X,lin_reg2.predict(X_poly), color='red')
plt.title('Polynomial Regression')
plt.xlabel('Yeras')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with linear regression
lin_reg2.predict(poly.fit_transform(6.5))