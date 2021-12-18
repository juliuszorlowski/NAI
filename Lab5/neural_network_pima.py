import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import KNNImputer
from sklearn.neural_network import MLPClassifier

"""
Opracowanie:
    Autorzy: Jakub Prucnal
             Juliusz Orłowski
    Temat:   Sieci Neuronowe dla Klasyfikacji
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
    Program wykorzystuje MLPClassifier z Solver sgd (odnosi się do stochastycznego zejścia gradientowego) w celu klasyfikacji danych do dwóch zbiorów:
        1. Nie zagrożony wystąpieniem cukrzycy
        2. Zagrożony wystąpieniem cukrzycy
Wykorzystywane biblioteki:
    Scikit-Learn - do uczenia sieci neuronowych i predykcji - oraz do uzupełnienia danych w dataset (KNNImputer)
    matplotlib - do tworzenia wizualizacji danych
    pandas - do analizy danych
    NumPy - do tworzenia macierzy

Dokumentacja kodu źródłowego:
    Python -> docstring (https://www.python.org/dev/peps/pep-0257/)
    matplotlib -> https://matplotlib.org/
    pandas -> https://pandas.pydata.org/
    NumPy -> https://numpy.org/doc/stable/user/whatisnumpy.html

"""

def dataset(data):
    data = pd.DataFrame(data)

input_file = 'pima-indians-diabetes.csv'
data = pd.read_csv(input_file, header=None)

# Cleaning dataset with kNN-Imputer
# Replace 0 -> Null
data[1:5] = data[1:5].replace(0, np.NaN)

X, y = data.loc[1:, 0:7], data.loc[1:, 8:]

# Using k-NN imputer replace NaN -> kNNValue
knn = KNNImputer()
knn.fit(X)
new_X = knn.transform(X)
new_X = pd.DataFrame(new_X)

# Scaling
new_X = preprocessing.minmax_scale(new_X)
new_X = pd.DataFrame(new_X)

# Return a flattened array
y = y.astype(int).values
y = y.ravel()

# Train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train, y_train = new_X[:num_training], y[:num_training]

# Test data
X_test, y_test = new_X[num_training:], y[num_training:]

# Neural Network Classifier - 2 hidden layers with sizes: 20 and 9 - najlepszy wynik score
mlp = MLPClassifier(
    hidden_layer_sizes=(20, 9,),
    max_iter=15,
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.1,
)

# Catching warnings from MLPClassifier
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

    # Here is a kind of Magic of Machine Learning
    mlp.fit(X_train, y_train)

# Printing score for train and test dataset.
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
