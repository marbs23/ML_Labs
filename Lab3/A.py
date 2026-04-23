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

    SSm = np.sum((y - np.mean(y))**2)
    historial = []
    for i in range(iteraciones):
        mse = calcular_mse(X, y, W)
        if (i > 0 and abs(mse-historial[-1][2])< 0.000001):
            historial.append((i, W.copy(), mse))
            break
        historial.append((i, W.copy(), mse))
        grad_W = gradiente_mse(X, y, W)
        W = W - lr * grad_W
    SSr = mse * len(y)
    R2 = 1 - SSr/SSm
    return W.copy(), historial, R2

def normal_ecuation(data):
    data_array = np.array(data)
    X_init = data_array[:, :-1]
    col_1 = np.ones((len(data), 1))
    X = np.hstack((col_1, X_init))
    y = data_array[:, -1:]
    
    # Aplicar la fórmula: (X^T * X)^-1 * X^T * y
    W_analitico = np.linalg.inv(X.T @ X) @ (X.T @ y)    
    return W_analitico

def data_log(data):
    new_data = [[math.log(row[0]), row[1]] for row in data]
    return new_data

def graficar(data, W, W_normal,  label_x, label_y, case, mode):
    dirname = f"outputs/{case}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    x_vals = [d[0] for d in data]
    y_vals = [d[1] for d in data]
    plt.figure(figsize=(10, 5))
    plt.scatter(x_vals, y_vals, color='blue', label='Data')
    x_line = np.array([min(x_vals), max(x_vals)])
    y_line_gd = W[0][0] + W[1][0] * x_line
    plt.plot(x_line, y_line_gd, color='red', linewidth=3, 
             label=f'GD: y={W[1][0]:.2f}x + {W[0][0]:.2f}')
    y_line_norm = W_normal[0][0] + W_normal[1][0] * x_line
    plt.plot(x_line, y_line_norm, color='green', linestyle='--', linewidth=2,
             label=f'Normal Ec.: y={W_normal[1][0]:.2f}x + {W_normal[0][0]:.2f}')
    plt.title(f'Visualización del Modelo {mode} {case}')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dirname}/{label_x}_{mode}.png")
    print(f"Save: {label_x} with {mode}.png")
    plt.close()

def print_hist(hist):
    for i in range(0,len(hist), 200):
        iteration, pesos, mse = hist[i]
        print(f"Iter {iteration:4d} | MSE: {mse:.8f} | b: {pesos[0][0]:.4f} | w: {pesos[1][0]:.4f}")    
    if len(hist) % 200 != 1:
        iteration, pesos, mse = hist[-1]
        print(f"Iter {iteration:4d} | MSE: {mse:.8f} | b: {pesos[0][0]:.4f} | w: {pesos[1][0]:.4f} (Final)")
    

# MAIN
if __name__ == "__main__":
    new_data = data_log(data)
    w_GD, hist_GD, R2 = gradient_descent(new_data, lr=0.05, iteraciones=2000)
    w_normal = normal_ecuation(new_data)
    print(f"Final Weights and R2: b={w_GD[0][0]:.4f}, w={w_GD[1][0]:.4f}, R2 = {R2}")
    print(f"Normal Ecuation: b={w_normal[0][0]:.4f}, w={w_normal[1][0]:.4f}")
    print(f"-------")
    print_hist(hist_GD)
    graficar(new_data, w_GD, w_normal, "Distancia", "Nivel de Ruido", "A", "GD")