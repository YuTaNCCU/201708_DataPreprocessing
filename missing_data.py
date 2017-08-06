# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')   #need setworking folder first 
X = dataset.iloc[:, :-1].values    #Ｘ＝first column to second last column
Y = dataset.iloc[:, 3].values      #Y= the last column
print('X=',X)
print('Y=',Y)  

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #set the parameter of Imputer
#imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.fit_transform(X[:, 1:3]) #choose the range which we want to impute