import pandas as pd
import numpy as np
from tabulate import tabulate

data = pd.read_csv('Iris.csv')
data = data.drop(['Id', 'Species'], axis=1)

sample_data = data.sample(n=130, random_state=0, replace=False)
X = sample_data[['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = sample_data['SepalLengthCm']
np.random.seed(0)
indices = np.random.permutation(len(X))
X_train, X_test = X.iloc[indices[:104]], X.iloc[indices[104:124]]
y_train, y_test = y.iloc[indices[:104]], y.iloc[indices[104:124]]
X_train.insert(0, 'Interceptada', 1)
coefficients = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

results_list = []

for i in range(len(X_test)):
    nueva_flor = X_test.iloc[i].values  
    prediccion_nueva_flor = np.insert(nueva_flor, 0, 1).dot(coefficients)
    
    error_estandar = y_test.iloc[i] - prediccion_nueva_flor
    mse = error_estandar**2
    precision = 1 - (mse / (y_test.iloc[i] - np.mean(y_test))**2)
    raiz_de_error_cuadratico = np.sqrt(mse)
    scr = (prediccion_nueva_flor - np.mean(y_test))**2

    results_list.append({
        'Predicción': prediccion_nueva_flor,
        'Error Estandar': error_estandar,
        'MSE': mse,
        'Precision': precision,
        'Raiz de error cuadratico': raiz_de_error_cuadratico,
        'SCR': scr
    })

sst = np.sum((y_test - np.mean(y_test))**2)  
sse = np.sum([result['MSE'] for result in results_list])  

error_estandar_regresion = np.sqrt(sse / len(X_test))  
r2 = 1 - (sse / sst)  

results_list.append({
    'Predicción': 'Resultado Final',
    'Error Estandar': error_estandar_regresion,
    'MSE': sse,
    'Precision': r2,
    'Raiz de error cuadratico': '',
    'SCR': sse
})

results_df = pd.DataFrame(results_list)

print(tabulate(results_df, headers='keys', tablefmt='pretty'))

print(f'\nError Estándar de la Regresión: {error_estandar_regresion:.4f}')
print(f'Coeficiente de Determinación (R^2): {r2:.4f}')