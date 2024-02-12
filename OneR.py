import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

Datos = pd.read_csv("Iris.csv", encoding='latin1', low_memory=False)
Datos = Datos.drop(['Id'], axis=1)

Prueba = Datos.sample(n=15, random_state=42)
ClaseP = Prueba.iloc[:15, 4]

Datos = Datos.drop(Prueba.index, axis=0)
Clase = Datos.iloc[:, 4]

Datos = Datos.drop(['Species'], axis=1)
Prueba = Prueba.drop(['Species'], axis=1)

X = Datos.to_numpy()
Y = Clase.to_numpy()

XP = Prueba.to_numpy()
YP = ClaseP.to_numpy()

def one_r(X, Y):
    best_feature = None
    best_accuracy = 0.0
    best_rules = None
    worst_feature = None  
    worst_accuracy = 1.0  
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        accuracy = 0
        rules = {}
        for value in values:
            mask = X[:, feature] == value
            predicted_class = np.argmax(np.bincount(Y[mask]))
            accuracy += np.sum(Y[mask] == predicted_class)
            rules[value] = predicted_class
        accuracy /= len(Y)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_feature = feature
            best_rules = rules
        if accuracy < worst_accuracy:
            worst_accuracy = accuracy
            worst_feature = feature
    return best_feature, best_rules, worst_feature

feature_idx, rules, least_relevant_feature = one_r(X, Y)
print("Índice de la característica más relevante:", feature_idx)
print("Índice de la característica menos relevante:", least_relevant_feature)

print("Reglas de predicción:")
for value, predicted_class in rules.items():
    print(f"  - Si '{Datos.columns[feature_idx]}' = {value}, entonces la clase es '{predicted_class}'")

predicted_class = np.argmax(np.bincount(Y))
y_pred = np.full_like(YP, predicted_class)
test_acc = np.mean(y_pred == YP)
print(f'Exactitud en el conjunto de prueba: {test_acc * 100:.2f}%')

error_total = 1 - test_acc
print(f'Error Total: {error_total * 100:.2f}%')

print("Etiquetas de los Datos de Entrenamiento:")
print(Y)

train_acc = np.mean(np.full_like(Y, predicted_class) == Y)
print(f'Exactitud en los Datos de Entrenamiento: {train_acc * 100:.2f}%')
