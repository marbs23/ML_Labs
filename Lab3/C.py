import numpy as np

dataset = [
    [256.0, 15.2, 990.8],
    [783.0, 27.4, 1015.5],
    [490.0, 31.4, 1268.4],
    [346.0, 33.0, 1226.4],
    [519.0, 34.7, 1226.5],
    [371.0, 24.9, 1283.1],
    [718.0, 22.0, 1168.1],
    [225.0, 32.3, 1063.9],
    [265.0, 22.9, 1088.2],
    [660.0, 27.9, 1207.8],
    [231.0, 15.4, 904.1],
    [665.0, 31.2, 1212.8]
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
        if (i > 0 and abs(mse - historial[-1][2]) < 0.01):
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
    X1 = data_array[:, :-1]
    X2 = X1**2
    y = data_array[:, -1:]
    result = np.hstack((X1,X2,y))
    return result.tolist()

def print_hist(hist):
    for i in range(0,len(hist), 100):
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
    normal_dataset = Zscore(dataset)
    data_norm_transform = Zscore(data_transform)

    w_GD, hist_GD, R2 = gradient_descent(normal_dataset, lr=0.2, iteraciones=2000)
    w_GDtrans, hist_GDtrans, R2trans = gradient_descent(data_norm_transform, lr=0.2, iteraciones=2000)
    w_normal = normal_equation(normal_dataset)
    w_normal_trans = normal_equation(data_norm_transform)
    w_normal_trans_real = normal_equation(data_transform)

    print("Linear Model:")
    print(f"GD: b={w_GD[0][0]:.4f}, w={w_GD.flatten()[1:]}, R2={R2:.6f}")
    print(f"Normal: b={w_normal[0][0]:.4f}, w={w_normal.flatten()[1:]}")
    print_hist(hist_GD)

    print("\nModel new Features (Cuadratic)")
    print(f"GD: b={w_GDtrans[0][0]:.4f}, w={w_GDtrans.flatten()[1:]}, R2={R2trans:.6f}")
    print(f"Normal: b={w_normal_trans[0][0]:.4f}, w={w_normal_trans.flatten()[1:]}")
    print_hist(hist_GDtrans)

    print(f"\nR2 Comparisson:")
    print(f"Linear: R2 = {R2:.6f}")
    print(f"Cuadratic: R2 = {R2trans:.6f}")

    w_aux = w_normal_trans_real.flatten()
    agua_opt = -w_aux[1] / (2 * w_aux[3])
    temp_opt  = -w_aux[2] / (2 * w_aux[4])
    print(f"Agua óptima: {agua_opt:.2f} L/ha/día")
    print(f"Temp óptima: {temp_opt:.2f} °C")