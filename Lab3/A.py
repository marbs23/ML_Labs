import random
import matplotlib.pyplot as plt
import numpy as np
import math
import os

data = [[1, 119.9],
        [2,106.0],
        [5, 89.4],
        [10, 72.1],
        [20, 61.2],
        [50, 39.2],
        [100, 27.6],
        [200, 13.7],
        [500, -2.6],
        [1000, -18.2]]

def predecir(X, W):
    return X @ W

def calcular_mse(X, y, W):
    y_pred = predecir(X, W)
    mse = np.mean((y - y_pred)**2)
    return mse

def gradiente_mse(X, y, W):
    N = len(y)
    y_pred = predecir(X, W)
    error = y - y_pred
    grad_W = (-2 / N) * (X.T @ error)
    return grad_W

# GRADIENT DESCENT (GD)
def gradient_descent(datos, lr=0.01, iteraciones=1000):
    datos_array = np.array(datos)
    X_init = datos_array[:, :-1]
    col_1 = np.ones((len(datos), 1))
    X = np.hstack((col_1, X_init))

    y = datos_array[:, -1:]
    W = np.zeros((X.shape[1], 1))
    historial = []
    for i in range(iteraciones):
        mse = calcular_mse(X, y, W)
        if (i > 0 and abs(mse-historial[-1][2])< 0.00001):
            historial.append((i, W.copy(), mse))
            break
        historial.append((i, W.copy(), mse))
        grad_W = gradiente_mse(X, y, W)
        W = W - lr * grad_W
    return W.copy(), historial

def data_log(data):
    new_data = [[math.log(row[0]), row[1]] for row in data]
    return new_data

def graficar(data, W, label_x, label_y, case, mode):
    dirname = f"outputs/{case}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    x_vals = [d[0] for d in data]
    y_vals = [d[1] for d in data]
    plt.figure(figsize=(10, 5))
    plt.scatter(x_vals, y_vals, color='blue', label='Data')
    x_line = np.array([min(x_vals), max(x_vals)])
    y_line = W[0][0] + W[1][0] * x_line
    plt.plot(x_line, y_line, color='red', linewidth=3, label=f'Modelo: y={W[0][0]:.4f} + {W[1][0]:.4f}x')
    plt.title(f'Visualización del Modelo {mode} {case}')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dirname}/{label_x}_{mode}.png")
    print(f"Save: {label_x} with {mode}.png")
    plt.close()

def print_hist(hist):
    for i in range(5):
        iteration, pesos, mse = hist[i]
        print(f"Iter {iteration:04d} | MSE: {mse:.8f} | b: {pesos[0][0]:.4f} | w: {pesos[1][0]:.4f}")
    
    print("...")
    
    # Imprimimos las últimas 5 épocas
    for i in range(-5, 0):
        iteration, pesos, mse = hist[i]
        print(f"Iter {iteration:04d} | MSE: {mse:.8f} | b: {pesos[0][0]:.4f} | w: {pesos[1][0]:.4f}")
    

# MAIN
if __name__ == "__main__":
    new_data = data_log(data)
    w_GD, hist_GD = gradient_descent(new_data, lr=0.01, iteraciones=2000)
    print(f"Pesos finales: Intercepto (b)={w_GD[0][0]:.4f}, Pendiente (w)={w_GD[1][0]:.4f}")
    print(f"-------")
    print_hist(hist_GD)
    graficar(new_data, w_GD, "Distancia", "Nivel de Ruido", "A", "GD")