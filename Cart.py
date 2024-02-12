import pandas as pd
import numpy as np

data = pd.read_csv('Iris.csv')

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = data[features]
y = data['Species']

np.random.seed(0)  
test_indices = np.random.choice(len(X), 20, replace=False)
train_indices = np.array(list(set(range(len(X))) - set(test_indices)))

X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

correlation_sepal = X_train[['SepalLengthCm', 'SepalWidthCm']].corr()
correlation_petal = X_train[['PetalLengthCm', 'PetalWidthCm']].corr()

print("Correlación entre las columnas Sepal:")
print(correlation_sepal)
print("\nCorrelación entre las columnas Petal:")
print(correlation_petal)

print("Datos de entrenamiento:")
print(X_train)
print("\nDatos de prueba:")
print(X_test)

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  
        self.threshold = threshold  
        self.left = left  
        self.right = right  
        self.value = value  

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def split_data(X, y, feature, threshold):
    left_mask = X.iloc[:, feature] <= threshold
    right_mask = ~left_mask
    return left_mask, right_mask

def find_best_split(X, y):
    num_features = X.shape[1]
    min_mse = float('inf')
    best_split = None
    
    for feature in range(num_features):
        thresholds = np.unique(X.iloc[:, feature])
        for threshold in thresholds:
            left_mask, right_mask = split_data(X, y, feature, threshold)
            if len(y[left_mask]) > 0 and len(y[right_mask]) > 0:
                y_pred = np.concatenate((np.repeat(np.mean(y[left_mask]), len(y[left_mask])), 
                                         np.repeat(np.mean(y[right_mask]), len(y[right_mask]))))
                mse = mean_squared_error(y, y_pred)
                if mse < min_mse:
                    min_mse = mse
                    best_split = (feature, threshold)
    
    return best_split

def build_tree(X, y, depth=0, max_depth=None):
    if depth == max_depth or len(set(y)) == 1:
        return TreeNode(value=np.mean(y))
    
    feature, threshold = find_best_split(X, y)
    if feature is not None:
        left_mask, right_mask = split_data(X, y, feature, threshold)
        left_tree = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
        right_tree = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)
        return TreeNode(feature=feature, threshold=threshold, left=left_tree, right=right_tree)
    else:
        return TreeNode(value=np.mean(y))

regression_tree = build_tree(X_train, y_train, max_depth=3)

def predict(tree, X):
    if tree.feature is None:
        return tree.value
    else:
        if X[tree.feature] <= tree.threshold:
            return predict(tree.left, X)
        else:
            return predict(tree.right, X)
        
predicted_classes = []
for _, row in X_test.iterrows():
    predicted_class = predict(regression_tree, row)
    predicted_classes.append(predicted_class)

mse_test = mean_squared_error(y_test, predicted_classes)

print("Clases Predichas:")
print(predicted_classes)
print("\nClases Reales:")
print(y_test.values)
print("\nError Cuadrático Medio para Datos de Prueba:", mse_test)

def print_tree(node, spacing=""):
    if node is None:
        return

    if node.feature is not None:
        print(spacing + "Característica:", features[node.feature])
        print(spacing + "Umbral:", node.threshold)
        left_mask, right_mask = split_data(X_train, y_train, node.feature, node.threshold)
        left_y, right_y = y_train[left_mask], y_train[right_mask]
        mse_left = mean_squared_error(left_y, np.full_like(left_y, np.mean(left_y)))
        mse_right = mean_squared_error(right_y, np.full_like(right_y, np.mean(right_y)))
        print(spacing + "Error Izquierda:", mse_left)
        print(spacing + "Error Derecha:", mse_right)
        print(spacing + "--> Izquierda:")
        print_tree(node.left, spacing + "  ")
        print(spacing + "--> Derecha:")
        print_tree(node.right, spacing + "  ")
    else:
        print(spacing + "Valor:", node.value)  

print("\nEstructura del árbol de regresión:")
print_tree(regression_tree)
