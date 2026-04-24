import matplotlib.pyplot as plt
import numpy as np
import os

dataset = [
    [95, 4, 13, 171600],
    [123, 2, 14, 216200],
    [118, 2, 22, 196900],
    [96, 4, 8, 190200],
    [182, 4, 19, 305700],
    [86, 4, 5, 180100],
    [63, 2, 7, 133600],
    [193, 2, 3, 320000],
    [155, 2, 29, 242700],
    [128, 2, 3, 216800],
    [195, 5, 18, 357400],
    [115, 5, 22, 219800]
]

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
        if (i > 0 and abs(mse-historial[-1][2])< 0.01):
            historial.append((i, W.copy(), mse))
            break
        historial.append((i, W.copy(), mse))
        grad_W = gradiente_mse(X, y, W)
        W = W - lr * grad_W
    SSr = mse * len(y)
    R2 = 1 - SSr/SSm
    return W.copy(), historial, R2

def normal_equation(data):
    data_array = np.array(data)
    X_init = data_array[:, :-1]
    col_1 = np.ones((len(data), 1))
    X = np.hstack((col_1, X_init))
    y = data_array[:, -1:]
    
    W_analitico = np.linalg.inv(X.T @ X) @ (X.T @ y)    
    return W_analitico

def Zscore(data):
    data_array = np.array(data, dtype=float)
    X = data_array[:, :-1]
    y = data_array[:, -1:]
    means = np.mean(X, axis=0)
    sds = np.std(X, axis=0)    
    X_normalized = (X - means) / sds    
    result = np.hstack((X_normalized, y))
    return result.tolist()

def graphicConvergence(hist, case):
    dirname = f"outputs/{case}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    mse_data = [d[2] for d in hist]
    plt.figure(figsize=(10, 6))
    plt.plot(mse_data, color='red', linewidth=1, label=f"MSE data")
    plt.title("Convergence Graph")
    plt.xlabel("Iterations")
    plt.ylabel("Error MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dirname}/{case}_convergence.png")
    print(f"Save: Convergence graph with GD.png")
    plt.close()

def graphicComparisson(w_GD, w_normal, normal_dataset, dataset, case):
    dirname = f"outputs/{case}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # Predicciones GD (datos normalizados)
    data_norm = np.array(normal_dataset, dtype=float)
    X_norm = np.hstack((np.ones((len(normal_dataset), 1)), data_norm[:, :-1]))
    y_real = data_norm[:, -1]
    y_pred_GD = predecir(X_norm, w_GD).flatten()

    # Predicciones Normal Equation (datos originales)
    data_orig = np.array(dataset, dtype=float)
    X_orig = np.hstack((np.ones((len(dataset), 1)), data_orig[:, :-1]))
    y_real_orig = data_orig[:, -1]
    y_pred_normal = predecir(X_orig, w_normal).flatten()

    plt.figure(figsize=(8, 6))    
    plt.scatter(y_real, y_pred_GD, color='blue', alpha=0.7, label='Predicciones')
    plt.scatter(y_real_orig, y_pred_normal, color='green', alpha=0.7, marker='^', label='Ec. Normal',)

    all_vals = np.concatenate([y_real, y_pred_GD, y_pred_normal])
    lims = [all_vals.min() * 0.97, all_vals.max() * 1.03]
    plt.plot(lims, lims, color='red', linestyle='--', label='Predicción Perfecta')

    plt.title(f'Comparación: Real vs Predicho - {case}')
    plt.xlabel('Precio Real (USD)')
    plt.ylabel('Precio Predicho (USD)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    filename = f"{dirname}/Comparisson_Real_Pred.png"
    plt.savefig(filename)
    print(f"Gráfica guardada en: {filename}")
    plt.close()

def print_hist(hist):
    for i in range(0,len(hist), 10):
        iteration, weights, mse = hist[i]
        weights_format = np.round(weights.flatten(), 4).tolist()
        print(f"Iter {iteration:4d} | MSE: {mse:.8f} | W: {weights_format}")    
    if len(hist) % 200 != 1:
        iteration, weights, mse = hist[-1]
        weights_format_final = np.round(weights.flatten(), 4).tolist()
        print(f"Iter {iteration:4d} | MSE: {mse:.8f} | W: {weights_format_final} (Final)")

# MAIN
if __name__ == "__main__":
    normal_dataset = Zscore(dataset)
    w_GD, hist_GD, R2 = gradient_descent(normal_dataset, lr=0.1, iteraciones=2000)
    w_normal = normal_equation(dataset)
    print("Linear Model:")
    print(f"GD: b={w_GD[0][0]:.4f}, w={w_GD.flatten()[1:]}, R2={R2:.6f}")
    print(f"Normal: b={w_normal[0][0]:.4f}, w={w_normal.flatten()[1:]}")
    print_hist(hist_GD)
    graphicConvergence(hist_GD, "B")
    graphicComparisson(w_GD, w_normal, normal_dataset, dataset, "B")