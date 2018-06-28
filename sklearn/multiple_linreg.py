

"""
A dataset of startup companies has data on 50 different startups in the following variables:
- Money spend on R&D
- Money spend on Administration
- Money spend on Marketing
- State
- Profit

This script fits a linear regression model to the data, to better understand patterns.
Data is in the data/50_Startups.csv.

The dataset and starting code were obtained from Udemy course "Machine Learning A-Z™: Hands-On Python & R In Data
Science" by Kirill Eremenko and Hadelin de Ponteves, link: https://www.udemy.com/machinelearning/
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the categorical variables into numeric
# This must be done before the split to train-test

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])

# OneHotEncoder cannot be directly used on the categorical data (ValueError),
# that's why we first need to use LabelEncoder (above)
oh = OneHotEncoder(categorical_features=[3])  # encode the 3-rd column of X
X = oh.fit_transform(X).toarray()

# Remove one of the OneHot encoded columns to avoid the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)


# Predicting the test set results
y_pred = reg.predict(X_test)


# Quantify the regression fit
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_pred, y_test)  # 83502864.03


# Print model's coefs
print("Coefs:", reg.coef_)  # [-9.592e+02  6.993e+02  7.734e-01  3.288e-02  3.661e-02]


# Use Backward Elimination to build a better model
import statsmodels.formula.api as sm

# Add a column of 1s to set x0=1, in the formula y = b0 + b1*x1 + b2x2 + ... (earlier sklearn did this automatically)
X = np.append(arr=np.ones((50, 1)).astype('int'), values=X, axis=1)  # actually appending X to the new column

# We choose a significance level SL=0.05 to which we'll compare the p-values of each variable below.

# Create a matrix to contain the optimal set of variables X_opt
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

# Create a new Ordinary Least Squares model using the statsmodels library
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# The output of the above line showed that the p-value of x2 is highest among all variables (=0.99).
# Therefore, we'll remove it and fit the regressor without it
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Now x1 has the largest p-value (=0.94), remove it
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Now x4 has the largest p-value > SL, remove it
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Conclusion: Only one variable has significant level of impact on predicting the pforit: R&D spent.
# Let's go back to sklearn and fit a linear model using only the R&D input data

reg_opt = LinearRegression()
X_train_opt = X_train[:, 2].reshape((len(X_train), 1))
X_test_opt = X_test[:, 2].reshape((len(X_test), 1))
reg_opt.fit(X_train_opt, y_train)
y_pred_opt = reg_opt.predict(X_test_opt)
mse_opt = mean_squared_error(y_pred_opt, y_test)  # 68473440

# We've reached a smaller Mean squared error (mse_opt < mse)



