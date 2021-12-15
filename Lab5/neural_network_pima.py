import warnings
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import KNNImputer
import sklearn.metrics as sm
from sklearn.neural_network import MLPClassifier

"""
Opracowanie:
    Autorzy: Jakub Prucnal
             Juliusz Orłowski
    Temat:   Uczenie SVM klasyfikowania danych
Wejście:
    - plik pima-indians-diabetes.csv zawierający zbiór danych z 768 badań medycznych przeprowadzonych na Indianach Pima
    pod kątem przewidywania wystąpienia cukrzycy w ciągu 5 lat od badania
    Dane zawierają:
        1. Ilość ciąż
        2. Poziom glukozy we krwi
        3. Rozkurczowe ciśnienie krwi (mm Hg)
        4. Grubość skóry nad tricepsem (mm)
        5. Ilość insuliny w surowicy (mu U/ml)
        6. Indeks BMI (waga w kg/(wzrost w m)^2)
        7. Funkcja cukrzycy genetycznej
        8. Wiek (w latach)
        9. Zmienna klasowa (0 lub 1)
Wyjście:
    Program wykorzystuje Support Vector Classificator w celu klasyfikacji danych do dwóch zbiorów:
        1. Nie zagrożony wystąpieniem cukrzycy
        2. Zagrożony wystąpieniem cukrzycy
Wykorzystywane biblioteki:
    NumPy - do tworzenia macierzy
    matplotlib - do analizy danych
    pandas - do tworzenia wizualizacji danych

Dokumentacja kodu źródłowego:
    Python -> docstring (https://www.python.org/dev/peps/pep-0257/)
    NumPy -> https://numpy.org/doc/stable/user/whatisnumpy.html
    matplotlib -> https://matplotlib.org/
    pandas -> https://pandas.pydata.org/
"""

def dataset(data):
    data = pd.DataFrame(data)

input_file = 'pima-indians-diabetes.csv'
data = pd.read_csv(input_file, header=None)

# Cleaning dataset with kNN-Imputer
# Replace 0 -> Null

X, y = data.loc[1:, 0:8], data.loc[1:, 8:]
print(data.loc[0])
print(y)
# Using k-NN imputer replace NaN -> kNNValue
knn = KNNImputer()
knn.fit(X)
new_X = knn.transform(X)
new_X = pd.DataFrame(new_X)


# Scaling
new_X = preprocessing.minmax_scale(new_X)
new_X = pd.DataFrame(new_X)


y = y.astype(int).values
y = y.ravel()


# Train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train, y_train = new_X[:num_training], y[:num_training]

# Test data
X_test, y_test = new_X[num_training:], y[num_training:]

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

mlp = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=10,
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.1,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))