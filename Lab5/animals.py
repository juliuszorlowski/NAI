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

input_file = 'cifar-10.csv'
data = pd.read_csv(input_file, header=None, skiprows=[0])

# Cleaning dataset with kNN-Imputer
# Replace 0 -> Null

X, y = data.loc[0:, :3072], data.loc[0:, 3072:]
X = X / 255.0
y = y.astype(int).values
y = y.ravel()
print(data.loc[0])

# Train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train, y_train = X[:num_training], y[:num_training]

# Test data
X_test, y_test = X[num_training:], y[num_training:]

mlp = MLPClassifier(
    hidden_layer_sizes=(1000, 800, 700, 600, 500, 400, 300, 200, 100, 50, 30, 20),
    max_iter=20,
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
