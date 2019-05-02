"""
Designed By: Akshaykanth D L
Date: 02-05-2019
Title : "Carbon Monoxide Prediction"


"""

import pandas as pd
from sklearn.svm import SVR
import numpy as numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#######################################################################################

#Fuel:- Petrol = 0, Diesel = 1

print("Data Pre Processing")
data = pd.read_csv("coEmission.csv", )
x = data.iloc[:,0:3].values
y = data.iloc[:, 3].values
x_tarin, y_train, x_test, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

print("Model Creating...!")
#svr_poly = SVR(kernel='rbf', C = 100, gamma = 'scale', degree=3, epsilon = 0.1, coef0=1, verbose = True)
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
pol_reg = LinearRegression()


print("Training Started..!")
pol_reg.fit(x_poly, y)
print("Training Ended")



while(True):

    print("##########################################################################################################################")
    print(" Year: ", end="")
    year = int(input()) - 2005
    print(" Temp: ", end="")
    temp = int(input())
    print(" Fuel: ", end="")
    fuel = int(input())
    params = [temp, fuel, year]
    print(" CO emission: ", pol_reg.predict(poly_reg.fit_transform([params]))[0], "grams")
 
