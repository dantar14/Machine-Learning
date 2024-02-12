import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def _region_query(self, data, point_idx):
        neighbors = []
        for i in range(len(data)):
            if self._distance(data[point_idx], data[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _expand_cluster(self, data, labels, point_idx, cluster_id, neighbors):
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            n_idx = neighbors[i]
            if labels[n_idx] == -1:  
                labels[n_idx] = cluster_id
            elif labels[n_idx] == 0:  
                labels[n_idx] = cluster_id
                new_neighbors = self._region_query(data, n_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            i += 1

    def fit(self, data):
        num_points = len(data)
        labels = [0] * num_points  
        cluster_id = 0

        for point_idx in range(num_points):
            if labels[point_idx] != 0:
                continue

            neighbors = self._region_query(data, point_idx)
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1  
            else:
                cluster_id += 1
                self._expand_cluster(data, labels, point_idx, cluster_id, neighbors)

        return labels
while True:
    eps = float(input("Ingresa el valor de epsilon (eps): "))
    min_samples = int(input("Ingresa el valor mínimo de puntos en un grupo (min_samples): "))
    file_path = "Iris.csv"  
    iris_data = pd.read_csv(file_path)
    iris_data = iris_data.drop(['Id', 'Species'], axis=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(iris_data)
    data = [tuple(row) for row in scaled_data]
    dbscan = DBSCAN(eps, min_samples)
    labels = dbscan.fit(data)
    iris_data['cluster'] = labels
    print("Resultados del algoritmo DBSCAN:")
    print(iris_data)
    print("Índices y Etiquetas:")
    for i, label in enumerate(labels):
        print(f"Índice {i}, Etiqueta: {label}")

    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap='viridis')
    plt.title('Resultados del DBSCAN')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.show()

    continuar = input("¿Deseas realizar otra operación? (s/n): ")
    if continuar.lower() != 's':
        break