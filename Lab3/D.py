import numpy as np

dataset = [
    [5, 2, 8, 16.6],
    [10, 4, 16, 21.2],
    [20, 5, 8, 30.1],
    [50, 8, 16, 38.5],
    [100, 10, 8, 50.5],
    [200, 12, 16, 50.8],
    [300, 8, 4, 58.6],
    [500, 15, 8, 66.4],
    [800, 10, 16, 64.8],
    [1000, 18, 4, 78.8],
    [1500, 12, 8, 74.4],
    [2000, 20, 16, 83.6]
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
        if (i > 0 and abs(mse-historial[-1][2])< 0.0001):
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

def transform_data(data):
    data_array = np.array(data, dtype=float)
    X1 = np.log(data_array[:, :1])
    y = data_array[:, -1:]
    result = np.hstack((X1,y))
    return result.tolist()

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
    data_transform = transform_data(dataset)
    data_norm = Zscore(data_transform)

    w_GD, hist_GD, R2 = gradient_descent(data_norm, lr=0.005, iteraciones=3000)
    w_normal = normal_equation(data_transform)

    print("Linear Model:")
    print(f"GD: b={w_GD[0][0]:.4f}, w={w_GD.flatten()[1:]}, R2={R2:.6f}")
    print(f"Normal: b={w_normal[0][0]:.4f}, w={w_normal.flatten()[1:]}")
    print_hist(hist_GD)
    #graphicConvergence(hist_GD, "B")
    #graphicComparisson(w_GD, normal_dataset, "B")