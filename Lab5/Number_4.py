import warnings
import pandas as pd
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.utils import column_or_1d

"""
Opracowanie:
    Autorzy: Jakub Prucnal
             Juliusz Orłowski
    Temat:   Sieci Neuronowe dla Klasyfikacji

Wejście:
    W celu uruchomienia programu należy pobrać plik csv ze strony https://www.openml.org/d/180 i przenieść go do
    katalogu głównego programu pod nazwą covertype.csv.

    - plik covertype.csv zawierający zbiór danych 110393 realnych środowisk leśnych komórkami po 30 na 30 metrów 
    oraz gatunek drzew przeważający na danym terenie.

    Kolumny: 
        Name / Data Type / Measurement / Description
        Elevation / quantitative /meters / Elevation in meters
        Aspect / quantitative / azimuth / Aspect in degrees azimuth
        Slope / quantitative / degrees / Slope in degrees
        Horizontal_Distance_To_Hydrology / quantitative / meters / Horz Dist to nearest surface water features
        Vertical_Distance_To_Hydrology / quantitative / meters / Vert Dist to nearest surface water features
        Horizontal_Distance_To_Roadways / quantitative / meters / Horz Dist to nearest roadway
        Hillshade_9am / quantitative / 0 to 255 index / Hillshade index at 9am, summer solstice
        Hillshade_Noon / quantitative / 0 to 255 index / Hillshade index at noon, summer solstice
        Hillshade_3pm / quantitative / 0 to 255 index / Hillshade index at 3pm, summer solstice
        Horizontal_Distance_To_Fire_Points / quantitative / meters / Horz Dist to nearest wildfire ignition points
        Wilderness_Area (4 binary columns) / qualitative / 0 (absence) or 1 (presence) / Wilderness area designation
        Soil_Type (40 binary columns) / qualitative / 0 (absence) or 1 (presence) / Soil Type designation
        Cover_Type (7 types) / integer / 1 to 7 / Forest Cover Type designation

    Etykiety danych:
        Etykieta 	Opis
        0 	        Spruce_fir
        1 	        Lodgepole_Pine
        2 	        Ponderosa_Pine
        3 	        Cottonwood_Willow
        4 	        Aspen
        5       	Douglas_fir
        6 	        Krummholz

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

input_file = 'covertype.csv'
data = pd.read_csv(input_file, header=None, skiprows=[0])

X, y = data.loc[0:, :53], data.loc[0:, 54:]

# Scaling X
X = preprocessing.minmax_scale(X)
X = pd.DataFrame(X)

# Encrypting labels for y
y = column_or_1d(y, warn=False)
encoder = preprocessing.LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)


# Train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train, y_train = X[:num_training], y[:num_training]

# Test data
X_test, y_test = X[num_training:], y[num_training:]

# Neural Network Classifier - 2 hidden layers with sizes: 256 and 64
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 64,),
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
