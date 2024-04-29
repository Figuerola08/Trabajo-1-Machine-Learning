import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def cargar_datos(ruta):
    """Carga los datos desde un archivo CSV."""
    data = pd.read_csv(ruta)
    return data

#Gonzalo Garcia
def analizar_datos(data):
    # Crear variables dummy
    dummieData = pd.get_dummies(data)
    
    # Calcular la matriz de correlación
    correlation_matrix = dummieData.corr()
    
    # Gráfica la matriz de correlación como un mapa de calor
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.show()

#Gonzalo Garcia
def preparar_datos(data):
    #Prepara los datos seleccionando las variables importantes y convirtiendo categorías en numérico.
    important_features = ['age', 'bmi', 'smoker']
    target = 'charges'
    data_selected = data[important_features + [target]].copy()
    data_selected.loc[:, 'smoker'] = data_selected['smoker'].map({'yes': 1, 'no': 0})
    return data_selected

#Diego Figuerola
def dividir_datos(data_selected):
    #Divide los datos en conjuntos de entrenamiento y prueba.
    x = data_selected[['age', 'bmi', 'smoker']]
    y = data_selected['charges']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#Diego Figuerola
def entrenar_modelo(X_train, y_train):
    #Entrena un modelo de regresión lineal.
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

#Diego Figuerola
def evaluar_modelo(model, X_test, y_test):
    #Evalúa el modelo usando el error cuadrático medio y calcula el RMSE.
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title('Real vs Predicted Values')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Línea de perfecta predicción
    plt.show()
    return mse, rmse

#Gonzalo Garcia
def imprimir_estadisticas(mse, rmse, charges_mean, charges_std):
    #Imprime el RMSE y las estadísticas descriptivas de 'charges'.
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"Media de 'charges': {charges_mean}")
    print(f"Desviación estándar de 'charges': {charges_std}")

def main():
    ruta = 'Medical_Cost.csv'
    data = cargar_datos(ruta)
    analizar_datos(data)
    data_preparada = preparar_datos(data)
    X_train, X_test, y_train, y_test = dividir_datos(data_preparada)
    modelo = entrenar_modelo(X_train, y_train)
    mse, rmse = evaluar_modelo(modelo, X_test, y_test)
    charges_mean = data['charges'].mean()
    charges_std = data['charges'].std()
    imprimir_estadisticas(mse, rmse, charges_mean, charges_std)

if __name__ == "__main__":
    main()
