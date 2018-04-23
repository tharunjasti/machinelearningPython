#importing libiraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#imorting dataset
df=pd.read_csv('50_Startups.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:, 3]  =labelencoder_X.fit_transform(X[:,3])

onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#avoiding Dummy variable trap
X=X[:,1:]


# Spliting the dataset into Training and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# Feature Scaling 
'''
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)'''

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result
y_pred=regressor.predict(X_test)

# Building the model using backword eliminatation
import statsmodels.formula.api as sm
#adding Xo variable i.e it is 1 or constant,  column to exting data  , dtype=np.int
X=np.append(arr=np.ones((50,1)).astype(int),values=X, axis=1 )
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

#removing with highest P-value column i.e no.2,
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

#removing with highest P-value column i.e no.1,
X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

#removing with highest P-value column i.e 4,
X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

#removing with highest P-value column i.e 5,
X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
