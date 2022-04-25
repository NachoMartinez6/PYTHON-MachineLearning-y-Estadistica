# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:30:08 2021

@author: Nacho
"""






# =============================================================================
# =============================================================================
"""                 --- LINEAR REGRESSION MODEL --- """
# =============================================================================
# =============================================================================










# =============================================================================
""" STEP 1: IMPORT OUR LIBRARIES """
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# =============================================================================
""" STEP 2: IMPORT OUR DATA"""
# =============================================================================

# price_data = pd.read_excel("oil_exxon.xlsx", engine="openpyxl")
price_data = pd.read_csv("oil_exxon.txt")


""" How to change one colummn for the index """


price_data.index = pd.to_datetime(price_data["date"])

price_data = price_data.drop("date", 1)



# # print(price_data.head())




# =============================================================================
""" STEP 3: CLEAN THE DATA"""
# =============================================================================

# # import missingno as msn

# # msn.matrix(price_data)
# # # msn.bar(price_data)
# # # msn.heatmap(price_data, figsize=(5,3), fontsize=(12))


# print(price_data.dtypes)
# print(price_data.isnull().sum())
# print(price_data.info())


""" Rename the column names """



# # columns = ["exxon_price", "oil_price"]
# # price_data.columns = columns


new_columns_names = {"exon_price": "exxon_price"}
price_data = price_data.rename(columns = new_columns_names)


# """ Get the exxon_price and oil_price mean """

  
# # print(price_data["exxon_price"].mean())
# # print(price_data["oil_price"].mean())

    
# # price_data["exxon_price"].replace(np.nan, 85, inplace=True)
# # price_data["oil_price"].replace(np.nan, 62, inplace=True)



""" Find the missing values """


# print(price_data.isnull().sum())
# print(price_data.info())
# print(price_data.isna().sum())




""" Remove the missing values """




price_data.dropna(axis=0, how="any", inplace=True)
# print(price_data.isna().sum())









# =============================================================================
"""          -- STEP 4: EXPLORE THE DATA -- """
# =============================================================================

""" We can visualize the relationship between the X and Y variables """


"""
BUILD A SCATTER PLOT
"""
""" Define the X and Y data """





# x = price_data["exxon_price"]
# y = price_data["oil_price"]


""" Make a scatter plot with the plot method """



# plt.scatter(x, y, c="cadetblue", marker="o", alpha=0.2, label="Daily Price")
# plt.show()

# # # plt.scatter(x, y, marker="o", color="cadetblue", alpha=0.5,label="Daily Price")


# plt.title("Exxon Vs. Oil")
# plt.xlabel("Exxon Mobile")
# plt.ylabel("Oil")

# plt.legend()
# plt.show()



""" MEASURE THE CORRELATION 

# Very strong relationship (|r|>0.8 =>)
# Strong relationship (0.6≤|r|)
# Moderate relationship (0.4≤|r|)
# Weak relationship (0.2≤|r|)
# Very weak relationship (|r|)

# """

# print(price_data.corr())

""" CREATE A STATISTICAL SUMMARY 

# FOR WHAT?

# To find OUTLIERS


# How we know if we have all our data into the distribution?

# If all the data falls within 3 Standard deviations of the mean
# """

# print(price_data.describe())




""" CHECKING FOR OUTLIERS AND SKEWNESS """


from scipy import stats
from scipy.stats import kurtosis,skew


""" Make a histogram to see the skew of the data """

# price_data.hist(grid=False, color="cadetblue")

# plt.figure()
# plt.subplot(1,2,1)
# plt.hist(X_price, color="cadetblue")

# plt.subplot(1,2,2)
# plt.hist(Y_price, color="cadetblue")


# plt.show()



""" Calculate the excess Kurtosis using the Fisher method """



# exxon_kurtosis = kurtosis(price_data["exxon_price"], fisher=True)
# oil_kurtosis = kurtosis(price_data["oil_price"], fisher=True)

# print("---------------")
# print("Exxon Kurtosis: {:.2f}".format(exxon_kurtosis))
# print("Oil Kurtosis: {:.2f}".format(oil_kurtosis))


# """ Calculate the skewness """

# exxon_skew = skew(price_data["exxon_price"])
# oil_skew = skew(price_data["oil_price"])


# print("--------------")
# print("Exxon Skew: {:.2f}".format(exxon_skew))
# print("Oil Skew: {:.2f}".format(oil_skew))





""" Create the kurtosis test """



# print("-- Exxon --")

# print(stats.kurtosistest(price_data["exxon_price"]))
# print(stats.skewtest(price_data["exxon_price"]))


# print("-- Oil --")


# print(stats.kurtosistest(price_data["oil_price"]))
# print(stats.skewtest(price_data["oil_price"]))



""" lOS TESTS nos dan que rechazariamos la normalidad en 3 de los 4
casos, incluso si los datos fueran ligeramente kurtosis o skewed. Esto es 
por lo que primero calculamos las metricas y visualizamos el dato antes de
realizar el test estadistico """



""" Si hubiera que transformar los datos ¿Como lo realizariamos? """












# =============================================================================
"""          -- STEP 5: BUILT THE MODEL  --             """
# =============================================================================


"""
Althought the data are slightly skewed  we are comfortable with de data, 
Nothing else is stopping us...

"""



""" Split the data """

from sklearn.model_selection import train_test_split


# X = price_data[["exxon_price"]].to_numpy()
# Y = price_data[["oil_price"]].to_numpy()


X = price_data[["oil_price"]]
Y = price_data[["exxon_price"]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)




""" Create and fit the model """

from sklearn import linear_model

model = linear_model.LinearRegression()

model.fit(X_train, y_train)



""" Explore the output """


""" En funcion de si lo hacemos mediante dataframes o arrays, vamos a tener
que añadir [0] o podremos usar el format de una forma u otra """

# print("PARAMETERS")
# print("------------------")

# print("Coefficient: ", model.coef_.round(2)[0][0])
# print("Intercept: ", model.intercept_.round(2)[0])








# """ Taking a single prediction """

# prediction = model.predict([[67.33]])


# print("Predictive value: ", 67.33)
# print("Prediction value: {:.2f}".format(prediction[0][0]))





""" Making multiple predictions """

from sklearn.metrics import r2_score


y_pred = model.predict(X_test)

# print("R2: {:.2f}".format(r2_score(y_test, y_pred)))





# =============================================================================
"""         -- STEP 6: EVALUATING THE MODEL --              """
# =============================================================================




import statsmodels.api as sm


""" Create a OLS model with statsmodels """

X2 = sm.add_constant(X)

model_sm = sm.OLS(Y, X2)

est = model_sm.fit()

# print(est.summary())



""" Confident intervals """

# print(est.conf_int())



""" Hypothesis Testing

Estimate the p-values 

NULL HYPOTHESIS: If it's more than 0.05, it means our coefficient equals 0

ALTERNATIVE HYPOTHESIS: If not the Coefficient does not equal 0

  """

# print(est.pvalues)



# """

# p-value is much less than 0.05 

# --> We reject the NULL HYPOTHESIS 

# """









# # =============================================================================
# """                 ---STEP 7: MODEL FIT ---                             """
# # =============================================================================



from sklearn.metrics import mean_squared_error, mean_absolute_error
# import math



# """ Metrics to calculate how our model fit to the data """



# print("Mean Squared Error (MSE): {:.2f}".format(mean_squared_error
#                                                 (y_test, y_pred)))


# print("Mean Absolute Error (MAE): {:.2f}".format(mean_absolute_error
#                                                   (y_test, y_pred)))


# print("Root Mean Squared Error (MSE): {:.2f}".format(math.sqrt
#                                                       (mean_squared_error
#                                                                 (y_test,y_pred))))



""" Plot the residuals """



# bins = np.arange(-18, 20, 4)
# res = (y_test - y_pred)

# # print(res.min())
# # print(res.max())


# plt.hist(res, color="royalblue", bins=bins)
# plt.xticks(bins)

# # (y_test - y_pred).hist(grid=False, color="royalblue")


# plt.title("- MODEL RESIDUALS -")
# plt.show()






""" Plot our line """


# plt.scatter(X_test, y_test, color="royalblue", alpha=0.3, label="Price")

# plt.plot(X_test, y_pred, "r-", linewidth=3, label="Linear Regression")


# plt.xlabel("Oil")
# plt.ylabel("Exxon Mobile")
# plt.title("Exxon Mobile Vs. Oil")


# plt.legend()
# plt.show()

# print("Oil Coefficient: {:.2f}".format(model.coef_[0][0]))
# print("Mean Squared Error (MSE): {:.2f}".format(mean_squared_error
#                                                 (y_test, y_pred)))
# print("R2: {:.2f}".format(r2_score
#                           (y_test, y_pred)))




# =============================================================================
"""         STEP 8: SAVE THE MODEL FOR FUTURE USE """
# =============================================================================




import pickle


""" pickle the model """

with open("linear_regression_model.sav", "wb") as f:
    pickle.dump(model,f)

""" load it back in"""

with open("linear_regression_model.sav", "rb") as picklefile:
    mdl_ex = pickle.load(picklefile)

""" Make a new prediction """

print(mdl_ex.predict([[67.33]]))



