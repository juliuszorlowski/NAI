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
data = pd.read_csv(input_file)

# Cleaning dataset with kNN-Imputer
# Replace 0 -> Null
data[['Glucose','BloodPressure','SkinThickness','Insuline','BMI']] = data[
    ['Glucose','BloodPressure','SkinThickness','Insuline','BMI']
    ].replace(0,np.NaN)

X, y = data.loc[:,[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insuline', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]], data.loc[:,['Outcome']]


# Using k-NN imputer replace NaN -> kNNValue
knn = KNNImputer()
knn.fit(X)
new_X = knn.transform(X)
new_X = pd.DataFrame(new_X)


# Scaling
new_X = preprocessing.minmax_scale(new_X)
new_X = pd.DataFrame(new_X)

# PCA transformation - Merge all columns in new_X to 2 colums.
X_pca = PCA(n_components=2).fit_transform(new_X)

y = y.astype(int).values
y = y.ravel()

svc = svm.SVC(kernel='rbf', C=10, gamma=1).fit(X_pca, y)

# create a mesh to plot in
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

h = sqrt(((x_max / x_min)/100)**2)

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))


# Predicted shape
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Drawing the plot
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c = y, marker='x')
plt.xlim(X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1)
plt.ylim(X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1)
plt.show()