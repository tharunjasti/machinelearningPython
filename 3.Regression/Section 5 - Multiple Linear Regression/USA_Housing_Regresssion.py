# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

housing=pd.read_csv('USA_Housing.csv')
housing.info()
housing.drop('Address',axis=1,inplace=True)
housing.info()

X=housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']].values
y=housing['Price'].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

plt.scatter(y_test,y_pred)
plt.show()

model.coef_

#statis for backword modeling  one way
## getting all p values for model
import statsmodels.api as sm
X2=sm.add_constant(X)
est=sm.OLS(y,X2).fit()
est.summary()


#checking backord eliminantion  2nd way
import statsmodels.formula.api as sms
X_sm=np.append(arr=np.ones((5000,1), dtype=np.int), values=X, axis=1 )

model_ols=sms.OLS(endog=y, exog=X_sm).fit()
model_ols.summary()
