import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

# DATA
def cargar_csv(ruta):
    list_data = []
    with open(ruta) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)
    for i in range(len(headers) - 1):
        datos = []
        for row in rows:
            x = float(row[headers[i]])
            y = float(row[headers[-1]])
            datos.append((x,y))
        list_data.append(datos)
    return headers, list_data


def predecir(x, w, b):
    return w * x + b

def calcular_mse(datos, w, b):
    n = len(datos)
    total = sum((y - predecir(x, w, b))**2  for x, y in datos)
    return total / n

def calcular_mae(datos, w, b):
    n = len(datos)
    total = sum(abs(y - predecir(x, w, b)) for x, y in datos)
    return total/n

def gradiente_mse(datos, w, b):
    n = len(datos)
    grad_w = 0.0
    grad_b = 0.0
    for x, y in datos:
        error = y - predecir(x, w, b)
        grad_w += -2 * x * error
        grad_b += -2 * error
    return grad_w / n, grad_b / n

def gradiente_una_muestra(x, y, w, b):
    error = y - predecir(x, w, b)
    grad_w = -2 * x * error
    grad_b = -2 * error
    return grad_w, grad_b

def solucion_analitica(datos):
    n = len(datos)
    sx  = sum(x for x, y in datos)
    sy  = sum(y for x, y in datos)
    sxy = sum(x*y for x, y in datos)
    sx2 = sum(x**2 for x, y in datos)

    m = (n * sxy - sx * sy) / (n * sx2 - sx**2)
    b = (sy - m * sx) / n
    return m, b

# GRADIENT DESCENT (GD)
def gradient_descent(datos, lr=0.01, iteraciones=1000,
                     verbose=True, mostrar_cada=100):
    w, b = 0.0, 0.0
    historial = []
    for i in range(iteraciones):
        mse = calcular_mse(datos, w, b)
        historial.append((i, w, b, mse))
        grad_w, grad_b = gradiente_mse(datos, w, b)
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w, b, historial

def stochastic_gradient_descent(datos, lr=0.01, epocas=100):
    w, b = 0.0, 0.0
    historial = []
    data_to_shuffle = datos.copy()
    for epoca in range(epocas):
        mse = calcular_mse(datos, w, b)
        historial.append((epoca, w, b, mse))
        random.shuffle(data_to_shuffle)
        for x_i, y_i in data_to_shuffle:
            grad_w, grad_b = gradiente_una_muestra(x_i, y_i, w, b)
            w = w - lr * grad_w
            b = b - lr * grad_b
    return w, b, historial

# ===========================================================================
# COMPARACION FINAL
# ===========================================================================

def graficar(datos, w, b, label_x, label_y, case, mode):
    dirname = f"outputs/{case}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    x_vals = [d[0] for d in datos]
    y_vals = [d[1] for d in datos]
    plt.figure(figsize=(10, 5))
    plt.scatter(x_vals, y_vals, color='blue', label='Data')
    x_line = np.array([min(x_vals), max(x_vals)])
    y_line = w * x_line + b
    plt.plot(x_line, y_line, color='red', linewidth=3, label=f'Modelo: y={w:.4f}x + {b:.4f}')
    plt.title("Visualización del Modelo "+ mode)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dirname}/{label_x}_{mode}.png")
    print(f"Save: {label_x} with {mode}.png")
    plt.close()

# MAIN
if __name__ == "__main__":

    print("LINEAR REGRESSION WITH GRADIENT DESCENT (UNIVARIABLE)")
    filename = "caso2_notas.csv"
    headers, data = cargar_csv(filename)
    for i in range(len(data)):
        # GD weights, bias and historial
        w_GD, b_GD, hist_GD = gradient_descent(data[i], lr=0.001, iteraciones=2000)

        # SGD weights, bias and historial
        w_SGD, b_SGD, hist_SGD = stochastic_gradient_descent(data[i], lr=0.001, epocas=200)

        graficar(data[i], w_GD, b_GD, headers[i], headers[-1], filename, "GD")
        graficar(data[i], w_SGD, b_SGD, headers[i], headers[-1], filename, "SGD")