import pandas as pd
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def calcular_num_hojas(birch):
    return birch.subcluster_labels_.max() + 1
continuar = True
while continuar:
    file_path = 'Iris.csv'
    data = pd.read_csv(file_path)
    data = data.drop(['Id', 'Species'], axis=1)
    threshold = float(input("Ingrese el umbral (threshold): "))
    branching_factor = int(input("Ingrese el factor de ramificación (branching_factor): "))
    X = data.values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    num_hojas = int(input("Ingrese el número de hojas deseado: "))
    umbral = None if num_hojas == -1 else num_hojas - 1  
    birch = Birch(n_clusters=None, threshold=umbral, branching_factor=branching_factor)
    print("Generando el árbol BIRCH:")
    birch.fit(X_pca)
    for i, subcluster_center in enumerate(birch.subcluster_centers_):
        print(f"Rama {i+1}: {subcluster_center}")
    cluster_labels = birch.predict(X_pca)
    data['Cluster'] = cluster_labels
    print("Resultado del clustering:")
    print(data)
    print("Etiquetas:")
    print(cluster_labels)
    print(f"Número de hojas configurado: {num_hojas}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', marker='o', s=50)
    plt.title("Resultado del Clustering BIRCH con PCA")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.colorbar(label='Clúster')
    plt.show()

    respuesta = input("¿Desea realizar otra operación? (s/n): ")
    if respuesta.lower() != 's':
        continuar = False
