# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Ecommerce Customers')

X=df[[ 'Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership' ]].values
y=df['Yearly Amount Spent'].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)



from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X=X_train, y=y_train)

y_pred=model.predict(X_test)


#checking backord eliminantion
import statsmodels.formula.api as sm
X_sm=np.append(arr=np.ones((500,1), dtype=np.int), values=X, axis=1 )

model_ols=sm.OLS(endog=y, exog=X_sm).fit()
model_ols.summary()

#removed with higehst p values i.e 3rd column
X_sm=X_sm[:,[0,1,2,4]]
model_ols=sm.OLS(endog=y, exog=X_sm).fit()
model_ols.summary()