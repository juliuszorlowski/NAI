import warnings
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

"""
Opracowanie:
    Autorzy: Jakub Prucnal
             Juliusz Orłowski
    Temat:   Sieci Neuronowe dla Klasyfikacji
    
Wejście:
    W celu uruchomienia programu należy pobrać plik csv ze strony https://www.openml.org/d/40927 i przenieść go do
    katalogu głównego programu pod nazwą cifar-10.csv.

    - plik cifar-10.csv zawierający zbiór danych 80 milionów małych obrazów wielkości 32x32 pikseli reprezentujących
    10 klas obiektów.
    
    Etykiety danych:
        Etykieta 	Opis
        0 	        airplane
        1 	        automobile
        2 	        bird
        3 	        cat
        4 	        deer
        5       	dog
        6 	        frog
        7       	horse
        8       	ship
        9 	        truck

Wyjście:
    Program wyświetla log z kolejnymi iteracjami uczenia oraz ostatecznym wynikiem treningowym i testowym.
    
Wykorzystywane biblioteki:
    pandas - do analizy danych
    matplotlib - do tworzenia wizualizacji danych

Dokumentacja kodu źródłowego:
    Python -> docstring (https://www.python.org/dev/peps/pep-0257/)
    matplotlib -> https://matplotlib.org/
    pandas -> https://pandas.pydata.org/
"""

input_file = 'cifar-10.csv'
data = pd.read_csv(input_file, header=None, skiprows=[0])


X, y = data.loc[0:, :3071], data.loc[0:, 3072:]
X = X / 255.0

# Return a flattened array
y = y.astype(int).values
y = y.ravel()

# Train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train, y_train = X[:num_training], y[:num_training]

# Test data
X_test, y_test = X[num_training:], y[num_training:]

# Neural Network Classifier - 3 hidden layers with sizes: 1000, 400 and 84
mlp = MLPClassifier(
    hidden_layer_sizes=(1000, 400, 84,),
    max_iter=20,
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
