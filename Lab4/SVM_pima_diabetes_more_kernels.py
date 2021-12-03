from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

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
Wykorzystane źródła:
https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
"""
def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


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

C = 1.0  # SVM regularization parameter
models = (
    svm.SVC(kernel="linear", C=C),
    svm.LinearSVC(C=C, max_iter=10000),
    svm.SVC(kernel="rbf", gamma=0.7, C=C),
    svm.SVC(kernel="poly", degree=3, gamma=40, C=C),
)
models = (clf.fit(X_pca, y) for clf in models)

# create a mesh to plot in
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
h = sqrt(((x_max / x_min)/100)**2)

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))



# title for the plots
titles = (
    "SVC with linear kernel",
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 3) kernel",
)

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_pca[:, 0], X_pca[:, 1]

# Drawing the plot
for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap='viridis', alpha=0.4)
    ax.scatter(X0, X1, c=y, cmap='viridis', s=10, marker='x')
    ax.set_xlim(X0.min() - 0.1, X0.max() + 0.1)
    ax.set_ylim(X1.min() - 0.1, X1.max() + 0.1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()