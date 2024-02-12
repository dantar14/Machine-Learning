import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tabulate import tabulate

data = pd.read_csv('Iris.csv')
data = data.drop(['Id', 'Species'], axis=1)

sample_data = data.sample(n=130, random_state=0, replace=False)
X = sample_data[['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = sample_data['SepalLengthCm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.insert(0, 'Interceptada', 1)
X_test.insert(0, 'Interceptada', 1)

coefficients = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

nueva_flor = np.array([1, 3.0, 5.1, 1.8])
prediccion_nueva_flor = nueva_flor.dot(coefficients)

y_pred = X_test.dot(coefficients)
error_prediccion = y_test.values - y_pred.values

print("Coeficientes:", coefficients)
print(f"Predicción para la nueva flor: {prediccion_nueva_flor:.2f}")
print("Error de predicción para las muestras de prueba:")
print(error_prediccion)

results_table = tabulate([coefficients], headers=X_train.columns.tolist() + ['Intercept'], tablefmt='pretty')
print("Coeficientes:")
print(results_table)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicciones')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Línea de Regresión')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Regresión Lineal Múltiple: Predicciones vs Valores Reales')
plt.legend()
plt.show()