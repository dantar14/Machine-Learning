import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def distancia_euclideana(punto1, punto2):
    return np.sqrt(np.sum((punto1 - punto2) ** 2))

datos = pd.read_csv('Iris.csv')
print("Dimensiones del DataSet:", datos.shape)
Indices = np.random.choice(150, 15, replace=False)
print("Indices:", Indices)

prueba = datos.iloc[Indices]
etiquetas = prueba['Species']
datos = datos.drop(Indices)

etiquetas2 = datos['Species']
datos = datos.drop(columns=['Id', 'Species'])

clase = prueba.iloc[:, 5]
prueba = prueba.drop(columns=['Id', 'Species'])

while True:
    try:
        K = int(input("Ingresa número de clústeres K: "))
    except ValueError:
        print("Debes escribir un número.")
        continue
    
    if K <= 0:
        print("Escribe un numero mayor que 0")
    else:
        break

arreglo = np.array(datos)
renglones = arreglo.shape[0]
columnas = arreglo.shape[1]
centroides = np.zeros((K, columnas))
centroides_nuevos = np.zeros((K, columnas))
cuenta = np.zeros(K)
distancia = np.zeros(K)

for i in range(K):
    centroides[i] = arreglo[np.random.choice(renglones)]

converge = False
iteracion = 0

while not converge:
    iteracion += 1
    for i in range(renglones):
        for j in range(K):
            distancia[j] = distancia_euclideana(arreglo[i], centroides[j])
        posicion = np.argmin(distancia)
        centroides_nuevos[posicion] += arreglo[i]
        cuenta[posicion] += 1
    
    for h in range(K):
        if cuenta[h] != 0:
            centroides_nuevos[h] /= cuenta[h]
    
    iguales = np.array_equal(centroides_nuevos, centroides)
    etiquetas2.reset_index(drop=True, inplace=True)
    
    print("Iteración:", iteracion)
    print("Centroides Actuales\n", centroides)
    print("Datos asignados por centroide =>", cuenta)
    print("Centroides Nuevos\n", centroides_nuevos)
    print("¿Converge? =>", iguales)
    input("Continuar...")

    if iguales:
        converge = True
    else:
        np.copyto(centroides, centroides_nuevos)
        cuenta.fill(0)
        centroides_nuevos.fill(0)

print("Grupos después de la última iteración:")
for i in range(renglones):
    distancia = [distancia_euclideana(arreglo[i], centroides[j]) for j in range(K)]
    posicion = np.argmin(distancia)
    print(f"Elemento {i}: Grupo {posicion}")
distancias_prueba = np.zeros((prueba.shape[0], K))

for i in range(prueba.shape[0]):
    for j in range(K):
        distancias_prueba[i, j] = distancia_euclideana(prueba.iloc[i], centroides[j])

etiquetas_prueba = np.argmin(distancias_prueba, axis=1)

for i in range(prueba.shape[0]):
    print(f"Punto de prueba {i}: Asignado al Grupo {etiquetas_prueba[i]} (Distancia a centroides: {distancias_prueba[i]}")

plt.scatter(centroides[:, 0], centroides[:, 1], c='black', marker='X', s=100, label='Centroides')
plt.scatter(arreglo[:, 0], arreglo[:, 1], c=etiquetas2.astype('category').cat.codes, cmap='cividis', alpha=0.5, s=50)
plt.legend()
plt.show()
