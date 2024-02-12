import numpy as np
import pandas as pd

Datos = pd.read_csv('Iris.csv', encoding='latin1', low_memory=False)
Datos = Datos.drop(['Id'], axis=1)
Clase = Datos.pop('Species').values
np.random.seed(0)  
indices = np.random.permutation(len(Datos))
x_train = Datos.iloc[indices[:-30]]  
x_test = Datos.iloc[indices[-20:]]  
y_train = Clase[indices[:-30]]
y_test = Clase[indices[-30:]]

Kvecinos = int(input("Ingrese el número de vecinos a comparar: "))

def distancia(punto1, punto2):
    return np.sqrt(np.sum((punto1 - punto2) ** 2))

for j in range(len(x_test)):
    Distancias = np.zeros((x_train.shape[0], 2))
    Distancias = Distancias.astype(float)

    k = 0
    for i in range(len(x_train)):
        Distancias[k][0] = distancia(x_train.iloc[i], x_test.iloc[j])
        Distancias[k][1] = y_train[i]
        k += 1

    Distancias = Distancias[Distancias[:, 0].argsort()]

    KNN = Distancias[0:Kvecinos, 1]

    print(f"Datos de prueba {j + 1} - Características: {x_test.iloc[j].values}, Clase real: {y_test[j]}")
    print("Distancias a los K vecinos más cercanos:")
    for i in range(Kvecinos):
        distancia_vecino = Distancias[i][0]
        vecino_idx = int(KNN[i])
        etiqueta_vecino = y_train[vecino_idx]
        print(f"Vecino {i + 1} - Distancia: {distancia_vecino:.2f}")

    votos = {}
    for clase in KNN:
        if clase in votos:
            votos[clase] += 1
        else:
            votos[clase] = 1

    clase_predicha = max(votos, key=votos.get)

    porcentaje_confianza = (votos[clase_predicha] / Kvecinos) * 100

    porcentaje_error = 100 - porcentaje_confianza

    print(f"Clase predicha: {clase_predicha}")
    print(f"Porcentaje de confianza: {porcentaje_confianza:.2f}%")
    print(f"Porcentaje de error: {porcentaje_error:.2f}%")
    print()
