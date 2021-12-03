from math import sqrt
from operator import ne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
# import sklearn.metrics as sm

"""
Opracowanie:
    Autorzy: Jakub Prucnal
             Juliusz Orłowski
    Temat:   Uczenie SVM klasyfikowania danych
Wejście:
    - plik gulls.csv zawierający zbiór danych z dokonanych pomiarów wymiarów ciała mew i określenia ich płci
    Dane zawierają:
        1. DCG - długość całkowita głowy od potylicy do końca dzioba
        2. WD - wysokość dzioba w najszerszym miejscu
        3. SzP - szpon - rozpiętość od końca ostatniego palca do początku pierwszego
        4. Skrz - długość skrzydła
        5. Sex - płeć
Wyjście:
    Program wykorzystuje Support Vector Classificator w celu klasyfikacji danych do dwóch zbiorów:
        1. Płeć żeńska
        2. Płeć męska
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

input_file = 'gulls.csv'
data = pd.read_csv(input_file)

X, y = data.loc[:,['DCG', 'WD', 'SzP', 'Skrz']], data.loc[:,['Sex']]


# Scaling
X = preprocessing.minmax_scale(X)
X = pd.DataFrame(X)

# PCA transformation - Merge all columns in new_X to 2 columns.
X_pca = PCA(n_components=2).fit_transform(X)

y = y.astype(int).values
y = y.ravel()

# Train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train, y_train = X_pca[:num_training], y[:num_training]

# Test data
X_test, y_test = X_pca[num_training:], y[num_training:]

svc = svm.SVC(kernel='rbf', C=1, gamma=50).fit(X_pca, y)

# create a mesh to plot in
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

h = sqrt(((x_max / x_min)/100)**2)

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))


# Predicted shape
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

y_test_pred = svc.predict(X_test)

# Drawing the plot
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c = y, marker='x')
plt.xlim(X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1)
plt.ylim(X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1)
plt.figure()

plt.contourf(xx, yy, Z, alpha=0.4)
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test, marker='x')
plt.xlim(X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1)
plt.ylim(X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1)
plt.legend(*scatter.legend_elements())

plt.show()
