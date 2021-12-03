from operator import ne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.impute import KNNImputer

input_file = './pima-indians-diabetes.csv'
data = pd.read_csv(input_file)

# Cleaning dataset with kNN-Imputer
# Replace 0 -> Null
data[['Glucose','BloodPressure','SkinThickness','Insuline','BMI']] = data[
    ['Glucose','BloodPressure','SkinThickness','Insuline','BMI']
    ].replace(0,np.NaN)

X, y = data.loc[:,[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insuline', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]], data.loc[:,['Outcome']]

print(data.head())
print(X.isnull().sum())
# Using k-NN imputer replace NaN -> kNNValue
knn = KNNImputer()
knn.fit(X)
new_X = knn.transform(X)
new_X = pd.DataFrame(new_X)

print(new_X.isnull().sum())

print(new_X.head())


# Scaling
new_X = preprocessing.minmax_scale(new_X)
new_X = pd.DataFrame(new_X)
print(new_X.head())

asfdsafdsa