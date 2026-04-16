"""
Regresion Lineal desde cero: Gradient Descent y Stochastic Gradient Descent
Dataset: Sugar Levels vs Blood Glucose Levels (UTEC - Linear Regression)
Autor: D.Sc. Manuel Eduardo Loaiza Fernandez
Observacion: implementacion pedagogica sin sklearn ni pytorch
"""

import math
import random


# ===========================================================================
# DATOS
# ===========================================================================

# Datos originales del slide
datos_originales = [
    (3.75, 13.06),
    (9.51, 24.62),
    (7.32, 17.72),
    (5.99, 13.46),
    (1.56, -3.49),
]

# Datos extendidos: incluye puntos adicionales estimados del grafico del slide
datos_extendidos = [
    (3.75, 13.06),
    (9.51, 24.62),
    (7.32, 17.72),
    (5.99, 13.46),
    (1.56, -3.49),
    # puntos adicionales estimados visualmente del grafico
    (0.5,   1.2),
    (0.8,  -1.5),
    (1.0,   2.0),
    (1.2,   0.5),
    (1.5,   3.1),
    (2.0,   4.5),
    (2.1,   1.8),
    (2.3,   8.0),
    (2.5,   5.2),
    (2.8,   6.0),
    (3.0,   9.5),
    (3.2,   4.8),
    (3.5,   8.3),
    (4.0,   9.0),
    (4.2,  13.0),
    (4.5,   8.5),
    (4.8,  11.0),
    (5.0,  12.5),
    (5.2,  10.0),
    (5.5,  13.5),
    (6.0,  14.0),
    (6.2,  15.5),
    (6.5,  12.0),
    (6.8,  16.0),
    (7.0,  15.0),
    (7.5,  18.5),
    (8.0,  17.0),
    (8.2,  25.0),
    (8.5,  16.5),
    (8.8,  17.5),
    (9.0,  20.0),
    (9.2,  11.0),
    (9.8,  22.0),
    (10.0, 31.0),
]


# ===========================================================================
# FUNCIONES BASE (sin librerias externas)
# ===========================================================================

def predecir(x, w, b):
    """f(x) = wx + b"""
    return w * x + b

def calcular_mse(datos, w, b):
    """MSE = (1/N) * sum((y - y_hat)^2)"""
    n = len(datos)
    total = sum((y - predecir(x, w, b))**2 for x, y in datos)
    return total / n

def calcular_mae(datos, w, b):
    """MAE = (1/N) * sum(|y - y_hat|)"""
    n = len(datos)
    total = sum(abs(y - predecir(x, w, b)) for x, y in datos)
    return total / n

def gradiente_mse(datos, w, b):
    """
    dJ/dw = (-2/N) * sum(x_i * (y_i - y_hat_i))
    dJ/db = (-2/N) * sum(y_i - y_hat_i)
    """
    n = len(datos)
    grad_w = 0.0
    grad_b = 0.0
    for x, y in datos:
        error = y - predecir(x, w, b)
        grad_w += -2 * x * error
        grad_b += -2 * error
    return grad_w / n, grad_b / n

def gradiente_una_muestra(x, y, w, b):
    """Gradiente sobre UNA sola muestra (para SGD)"""
    error = y - predecir(x, w, b)
    grad_w = -2 * x * error
    grad_b = -2 * error
    return grad_w, grad_b

def solucion_analitica(datos):
    """
    Soluccion exacta por minimos cuadrados:
    m = (N*sum(xy) - sum(x)*sum(y)) / (N*sum(x^2) - (sum(x))^2)
    b = (sum(y) - m*sum(x)) / N
    """
    n = len(datos)
    sx  = sum(x for x, y in datos)
    sy  = sum(y for x, y in datos)
    sxy = sum(x*y for x, y in datos)
    sx2 = sum(x**2 for x, y in datos)

    m = (n * sxy - sx * sy) / (n * sx2 - sx**2)
    b = (sy - m * sx) / n
    return m, b

def separador(titulo):
    linea = "=" * 60
    print(f"\n{linea}")
    print(f"  {titulo}")
    print(linea)


# ===========================================================================
# GRADIENT DESCENT (GD)
# ===========================================================================

def gradient_descent(datos, lr=0.01, iteraciones=1000,
                     verbose=True, mostrar_cada=100):
    """
    Batch Gradient Descent:
    En cada iteracion usa TODOS los datos para calcular el gradiente.

    Parametros:
        datos       : lista de tuplas (x, y)
        lr          : learning rate (alpha)
        iteraciones : numero maximo de pasos
        verbose     : imprimir progreso
        mostrar_cada: cada cuantas iteraciones imprimir
    """
    w, b = 0.0, 0.0   # inicializacion del slide: w=0, b=0
    historial = []

    if verbose:
        separador("GRADIENT DESCENT (GD)")
        print(f"  lr={lr}  |  N={len(datos)}  |  iter_max={iteraciones}")
        print(f"  {'Iter':>6}  {'w':>10}  {'b':>10}  {'MSE':>12}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}")

    for i in range(iteraciones):
        mse = calcular_mse(datos, w, b)
        historial.append((i, w, b, mse))

        if verbose and (i % mostrar_cada == 0 or i < 5):
            print(f"  {i:>6}  {w:>10.4f}  {b:>10.4f}  {mse:>12.4f}")

        # calcular gradiente con todos los datos
        gw, gb = gradiente_mse(datos, w, b)

        # actualizar parametros
        w = w - lr * gw
        b = b - lr * gb

    mse_final = calcular_mse(datos, w, b)
    historial.append((iteraciones, w, b, mse_final))

    if verbose:
        print(f"  {iteraciones:>6}  {w:>10.4f}  {b:>10.4f}  {mse_final:>12.4f}")
        print(f"\n  Resultado GD:  w = {w:.4f}  |  b = {b:.4f}")
        print(f"  MSE final   :  {mse_final:.4f}")
        print(f"  MAE final   :  {calcular_mae(datos, w, b):.4f}")

    return w, b, historial


# ===========================================================================
# STOCHASTIC GRADIENT DESCENT (SGD)
# ===========================================================================

def stochastic_gradient_descent(datos, lr=0.01, epocas=100,
                                  verbose=True, mostrar_cada=10):
    """
    Stochastic Gradient Descent (SGD):
    En cada paso usa UNA sola muestra aleatoria (shuffle por epoca).

    Parametros:
        datos      : lista de tuplas (x, y)
        lr         : learning rate
        epocas     : numero de pasadas completas por el dataset
        verbose    : imprimir progreso
        mostrar_cada: cada cuantas epocas imprimir
    """
    w, b = 0.0, 0.0
    historial = []
    random.seed(42)   # reproducibilidad

    if verbose:
        separador("STOCHASTIC GRADIENT DESCENT (SGD)")
        print(f"  lr={lr}  |  N={len(datos)}  |  epocas={epocas}")
        print(f"  {'Epoca':>6}  {'w':>10}  {'b':>10}  {'MSE':>12}  Nota")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*20}")

    for epoca in range(epocas):
        mse = calcular_mse(datos, w, b)
        historial.append((epoca, w, b, mse))

        nota = ""
        if epoca == 0:
            nota = "<-- inicio"
        elif verbose and epoca % mostrar_cada == 0:
            nota = ""

        if verbose and (epoca % mostrar_cada == 0 or epoca < 3):
            print(f"  {epoca:>6}  {w:>10.4f}  {b:>10.4f}  {mse:>12.4f}  {nota}")

        # SHUFFLE: mezclar datos al inicio de cada epoca
        indices = list(range(len(datos)))
        random.shuffle(indices)

        # recorrer cada muestra en orden aleatorio
        for idx in indices:
            x_i, y_i = datos[idx]
            gw, gb = gradiente_una_muestra(x_i, y_i, w, b)
            w = w - lr * gw
            b = b - lr * gb

    mse_final = calcular_mse(datos, w, b)
    historial.append((epocas, w, b, mse_final))

    if verbose:
        print(f"  {epocas:>6}  {w:>10.4f}  {b:>10.4f}  {mse_final:>12.4f}  <-- fin")
        print(f"\n  Resultado SGD: w = {w:.4f}  |  b = {b:.4f}")
        print(f"  MSE final    : {mse_final:.4f}")
        print(f"  MAE final    : {calcular_mae(datos, w, b):.4f}")

    return w, b, historial


# ===========================================================================
# DEMO DE PRIMERAS ITERACIONES (pedagogico)
# ===========================================================================

def demo_primeras_iteraciones(datos, lr=0.01, n_iter=5):
    separador(f"ITERACIONES MANUALES DETALLADAS (lr={lr})")
    w, b = 0.0, 0.0
    n = len(datos)

    for it in range(n_iter):
        print(f"\n--- Iteracion {it} ---")
        print(f"  w = {w:.6f},  b = {b:.6f}")

        mse = calcular_mse(datos, w, b)
        print(f"  MSE = {mse:.4f}")

        # mostrar errores individuales
        print(f"  {'x':>6}  {'y':>7}  {'y_hat':>8}  {'error':>8}  "
              f"{'x*error':>10}")
        sum_xerr = 0.0
        sum_err  = 0.0
        for x, y in datos:
            yhat  = predecir(x, w, b)
            err   = y - yhat
            sum_xerr += x * err
            sum_err  += err
            print(f"  {x:>6.2f}  {y:>7.2f}  {yhat:>8.4f}  {err:>8.4f}  "
                  f"{x*err:>10.4f}")

        gw = -2 * sum_xerr / n
        gb = -2 * sum_err  / n
        print(f"\n  grad_w = -2/N * {sum_xerr:.4f} = {gw:.6f}")
        print(f"  grad_b = -2/N * {sum_err:.4f}  = {gb:.6f}")

        w_new = w - lr * gw
        b_new = b - lr * gb
        print(f"\n  w <- {w:.6f} - {lr} * ({gw:.6f}) = {w_new:.6f}")
        print(f"  b <- {b:.6f} - {lr} * ({gb:.6f}) = {b_new:.6f}")
        w, b = w_new, b_new


# ===========================================================================
# COMPARACION FINAL
# ===========================================================================

def comparar_resultados(datos, nombre):
    separador(f"COMPARACION FINAL — {nombre}")

    # Solucion analitica (referencia exacta)
    w_analitica, b_analitica = solucion_analitica(datos)
    mse_analitica = calcular_mse(datos, w_analitica, b_analitica)
    print(f"\n  Solucion analitica (exacta):")
    print(f"    w = {w_analitica:.4f}  |  b = {b_analitica:.4f}  |  MSE = {mse_analitica:.4f}")

    # GD
    w_gd, b_gd, _ = gradient_descent(datos, lr=0.005, iteraciones=2000,
                                       verbose=False)
    mse_gd = calcular_mse(datos, w_gd, b_gd)
    print(f"\n  Gradient Descent (lr=0.005, 2000 iter):")
    print(f"    w = {w_gd:.4f}  |  b = {b_gd:.4f}  |  MSE = {mse_gd:.4f}")

    # SGD
    w_sgd, b_sgd, _ = stochastic_gradient_descent(datos, lr=0.005, epocas=200,
                                                    verbose=False)
    mse_sgd = calcular_mse(datos, w_sgd, b_sgd)
    print(f"\n  SGD (lr=0.005, 200 epocas):")
    print(f"    w = {w_sgd:.4f}  |  b = {b_sgd:.4f}  |  MSE = {mse_sgd:.4f}")

    print(f"\n  Diferencia GD  vs analitica: Δw={abs(w_gd-w_analitica):.4f},"
          f" Δb={abs(b_gd-b_analitica):.4f}")
    print(f"  Diferencia SGD vs analitica: Δw={abs(w_sgd-w_analitica):.4f},"
          f" Δb={abs(b_sgd-b_analitica):.4f}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    print("\n" + "#"*60)
    print("  REGRESION LINEAL DESDE CERO")
    print("  GD y SGD sin sklearn / pytorch")
    print("#"*60)

    # 1. Mostrar el efecto del learning rate del slide (a=0.1 diverge)
    separador("EFECTO DEL LEARNING RATE (datos originales)")
    print("\n  Con lr=0.1 (valor del slide) — primeras 3 iteraciones:")
    demo_primeras_iteraciones(datos_originales, lr=0.1, n_iter=3)

    print("\n\n  Con lr=0.005 (valor estable) — primeras 5 iteraciones:")
    demo_primeras_iteraciones(datos_originales, lr=0.005, n_iter=5)

    # 2. GD sobre datos originales
    gradient_descent(datos_originales, lr=0.005, iteraciones=2000,
                     mostrar_cada=400)

    # 3. SGD sobre datos originales
    stochastic_gradient_descent(datos_originales, lr=0.005, epocas=200,
                                 mostrar_cada=40)

    # 4. Comparacion sobre datos originales
    comparar_resultados(datos_originales, "DATOS ORIGINALES (N=5)")

    # 5. GD sobre datos extendidos
    gradient_descent(datos_extendidos, lr=0.005, iteraciones=2000,
                     mostrar_cada=400)

    # 6. SGD sobre datos extendidos
    stochastic_gradient_descent(datos_extendidos, lr=0.005, epocas=200,
                                 mostrar_cada=40)

    # 7. Comparacion sobre datos extendidos
    comparar_resultados(datos_extendidos, "DATOS EXTENDIDOS (N=35)")

    # 8. Predicciones finales con datos extendidos
    separador("PREDICCIONES FINALES (datos extendidos)")
    w_final, b_final = solucion_analitica(datos_extendidos)
    print(f"\n  Modelo: f(x) = {w_final:.4f}*x + ({b_final:.4f})")
    print(f"\n  {'Sugar Level':>12}  {'Prediccion':>12}  {'Descripcion'}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*20}")
    casos = [(2.0,"bajo"),(5.0,"medio"),(8.0,"alto"),(10.0,"muy alto")]
    for x, desc in casos:
        yhat = predecir(x, w_final, b_final)
        print(f"  {x:>12.1f}  {yhat:>12.2f}  {desc}")
