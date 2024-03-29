import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier

"""
Opracowanie:
    Autorzy: Jakub Prucnal
             Juliusz Orłowski
    Temat:   Sieci Neuronowe dla Klasyfikacji
    
Wejście:
    W celu uruchomienia programu należy pobrać plik csv ze strony https://www.openml.org/d/40996 i przenieść go do
    katalogu głównego programu pod nazwą fashion-db.csv.

    - plik fashion-db.csv z zestawem danych składających się z produktów Zalando - zestaw treningowy 60000 przykładów
    i zestaw testowy 10000 przykładów - obrazy 28x28 pikseli w skali odcieni szarości reprezentujących 10 klas
    produktów.
    
    Etykiety danych:
        Etykieta 	Opis
        0 	        T-shirt/top
        1 	        Trouser
        2 	        Pullover
        3 	        Dress
        4 	        Coat
        5       	Sandal
        6 	        Shirt
        7       	Sneaker
        8       	Bag
        9 	        Ankle boot
        
Wyjście:
    Program wyświetla log z kolejnymi iteracjami uczenia oraz ostatecznym wynikiem treningowym i testowym.
    Ponadto, program tworzy confusion matrix w celu ewaluacji skuteczności działania klasyfikatora.
    
Wykorzystywane biblioteki:
    pandas - do analizy danych
    matplotlib - do tworzenia wizualizacji danych

Dokumentacja kodu źródłowego:
    Python -> docstring (https://www.python.org/dev/peps/pep-0257/)
    matplotlib -> https://matplotlib.org/
    pandas -> https://pandas.pydata.org/
"""

input_file = 'fashion-db.csv'
data = pd.read_csv(input_file, header=None, skiprows=[0])

X, y = data.loc[0:, :783], data.loc[0:, 784:]
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

# Neural Network Classifier - 2 hidden layers with sizes: 120 and 84
mlp = MLPClassifier(
    hidden_layer_sizes=(120, 84,),
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

# Creating and visualisation the confusion matrix for predictions
predictions = mlp.predict(X_test)
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
cm = confusion_matrix(y_test, predictions, labels=mlp.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
disp.plot()
plt.show()
