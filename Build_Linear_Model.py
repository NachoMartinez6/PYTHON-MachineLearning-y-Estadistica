# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:58:54 2020

@author: Nacho
"""



"""
===============================================================
BUILD LINEAR REGRESSION MODEL IN PYTHON  (1º MODEL)
===============================================================
"""



"""         --- Load the diabetes dataset (via sklearn) ---      """

import pandas as pd 
import numpy as np

from sklearn import datasets

diabetes = datasets.load_diabetes()


""" Brief introduction to the data """

# print(diabetes)
# print(diabetes.DESCR)
# print(diabetes["feature_names"])

""" Create X and Y matrices """


# x_diabetes = diabetes["data"]
# y_diabetes = diabetes["target"]

# print(x.shape)
# print(y.shape)

""" Load dataset + Create X and Y (in 1 step) """


X_dbt, Y_dbt = datasets.load_diabetes(return_X_y=True)
# # print(x.shape, y.shape)





# import matplotlib.pyplot as plt
# X_dbt.hist(grid=False, color="blue")


# """         --- DATA SPLIT ---          """

from sklearn.model_selection import train_test_split



# """ Perform 80/20 Data split """

tr_x_dbt, te_x_dbt, tr_y_dbt, te_y_dbt = train_test_split(X_dbt, Y_dbt, test_size=0.2)








"""     --- LINEAR REGRESION  MODEL ---     """


""" Import library """


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


""" Build linear regression """

mdl_dbt = linear_model.LinearRegression()

mdl_dbt.fit(tr_x_dbt, tr_y_dbt)


""" Apply trained model to make predictionn (on test set)"""

pred_y_dbt = mdl_dbt.predict(te_x_dbt)








"""      --- Prediction Results ---            """



""" Print model performance """


print("MODEL PROPERTIES")
print("-----------------------")

print("Coefficients: ", mdl_dbt.coef_)
print("-----------------------")

print("Intercept: ", mdl_dbt.intercept_)
print("-----------------------")

print("Mean squared error (MSE): {:.2f}".format(mean_squared_error(te_y_dbt, pred_y_dbt)))
print("-----------------------")

print("Coefficient of determitation (R^2): {:.2f}".format(r2_score(te_y_dbt, pred_y_dbt)))
print("-----------------------")




""" to round a number in less floats!"""


# print("{:.2f}".format(r2_score(te_y_dbt, pred_y_dbt)))

# print(r2_score(te_y_dbt, pred_y_dbt).round(2))







""" Make Scatter Plot """


import seaborn as sns
import matplotlib.pyplot as plt


""" Plot the residuals """

# bins = np.arange(-161, 161, 40)

# te_y_dbt = pd.Series(te_y_dbt)
# pred_y_dbt = pd.Series(pred_y_dbt)

# (te_y_dbt - pred_y_dbt).hist(grid=False, color="royalblue", bins=bins)
# # (te_y_dbt - pred_y_dbt).hist(grid=False, color="royalblue")

# plt.title("Model Residuals")
# plt.show()






# # sns.scatterplot(x = Y_test, y = Y_pred, marker="+")
# sns.scatterplot(x = te_y_dbt, y = pred_y_dbt, alpha=0.5)

# plt.title("Comparisson between Test and Pred.")

# plt.show()




