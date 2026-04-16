"""
Regresion Lineal Multivariable — Graficas de convergencia
GD y SGD: learning rates por separado + comparacion
GD: inicializaciones por separado + comparacion
+ Graficas de pesos finales por escenario (barras)
"""

import math, random, csv, os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── carpeta de salida ──────────────────────────────────────────────────────
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── colores y estilo global ────────────────────────────────────────────────
C1, C2, C3 = '#2196F3', '#FF9800', '#E53935'   # azul / naranja / rojo
GRID_KW  = dict(alpha=0.2, linestyle='--', linewidth=0.8)
LINE_KW  = dict(linewidth=2.2, alpha=0.92)

FEATURE_NAMES = ['horas_estudio', 'horas_suenio', 'asistencia_pct',
                 'ejercicios_resueltos']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# ===========================================================================
# DATOS
# ===========================================================================

def cargar_csv(ruta):
    with open(ruta) as f:
        reader = csv.DictReader(f)
        datos = []
        for fila in reader:
            x = [float(fila['horas_estudio']), float(fila['horas_suenio']),
                 float(fila['asistencia_pct']), float(fila['ejercicios_resueltos'])]
            y = float(fila['nota_final ( Y )'])
            datos.append((x, y))
    return datos

def calcular_media_std(datos):
    n, p = len(datos), len(datos[0][0])
    medias = [sum(datos[i][0][j] for i in range(n)) / n for j in range(p)]
    stds   = [math.sqrt(sum((datos[i][0][j]-medias[j])**2 for i in range(n))/n)
              for j in range(p)]
    return medias, stds

def normalizar_zscore(datos, medias, stds):
    return [
        ([(x_j-mu)/(sigma if sigma>1e-8 else 1.0)
          for x_j,mu,sigma in zip(x,medias,stds)], y)
        for x, y in datos
    ]

# ===========================================================================
# MODELO
# ===========================================================================

def predecir(x, w, b):
    return sum(wi*xi for wi,xi in zip(w,x)) + b

def calcular_mse(datos, w, b):
    n = len(datos)
    return sum((y - predecir(x,w,b))**2 for x,y in datos) / n

def gradiente_mse(datos, w, b):
    n, p = len(datos), len(w)
    gw = [0.0]*p; gb = 0.0
    for x, y in datos:
        e = y - predecir(x, w, b)
        for j in range(p): gw[j] += -2*x[j]*e
        gb += -2*e
    return [g/n for g in gw], gb/n

def gradiente_una_muestra(x, y, w, b):
    e = y - predecir(x, w, b)
    return [-2*xj*e for xj in x], -2*e

def gradient_descent(datos, lr=0.01, iteraciones=500,
                     w_init=None, b_init=0.0):
    p = len(datos[0][0])
    w = list(w_init) if w_init else [0.0]*p
    b = b_init
    hist = []
    for _ in range(iteraciones):
        hist.append(calcular_mse(datos, w, b))
        gw, gb = gradiente_mse(datos, w, b)
        w = [wj - lr*gwj for wj,gwj in zip(w,gw)]
        b -= lr*gb
    hist.append(calcular_mse(datos, w, b))
    return w, b, hist

def stochastic_gradient_descent(datos, lr=0.01, epocas=100, seed=42):
    p = len(datos[0][0])
    w = [0.0]*p; b = 0.0
    hist = []
    random.seed(seed)
    for _ in range(epocas):
        hist.append(calcular_mse(datos, w, b))
        idx = list(range(len(datos))); random.shuffle(idx)
        for i in idx:
            xi, yi = datos[i]
            gw, gb = gradiente_una_muestra(xi, yi, w, b)
            w = [wj - lr*gwj for wj,gwj in zip(w,gw)]
            b -= lr*gb
    hist.append(calcular_mse(datos, w, b))
    return w, b, hist

# ===========================================================================
# HELPERS DE GRAFICAS
# ===========================================================================

def _fmt_ax(ax, title, xlabel, ylabel, log_y=False):
    ax.set_title(title, fontweight='bold', pad=7)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(**GRID_KW)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f'{v:,.0f}'))

def _anotar_final(ax, hist, color, log_y=False):
    """Pone un punto y etiqueta en el ultimo valor de la curva."""
    x_fin = len(hist) - 1
    y_fin = hist[-1]
    ax.scatter([x_fin], [y_fin], color=color, s=45, zorder=5)
    offset = (0, 6) if not log_y else (0, 0)
    ax.annotate(f'{y_fin:.1f}', xy=(x_fin, y_fin),
                xytext=offset, textcoords='offset points',
                fontsize=8, color=color, ha='right')

def _guardar(nombre):
    ruta = os.path.join(OUTPUT_DIR, nombre)
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  >> {ruta}')

# ===========================================================================
# NUEVA: GRAFICA DE PESOS FINALES (barras agrupadas)
# ===========================================================================

def grafica_pesos_finales(escenarios, titulo, nombre_archivo):
    """
    escenarios: lista de (label, color, w_final, b_final)
    Genera una figura con barras agrupadas: un grupo por feature + bias,
    una barra por escenario dentro de cada grupo.
    """
    import numpy as np

    n_features = len(escenarios[0][2])
    etiquetas  = FEATURE_NAMES[:n_features] + ['bias']
    n_vars     = len(etiquetas)
    n_escen    = len(escenarios)

    x      = np.arange(n_vars)
    ancho  = 0.8 / n_escen          # ancho de cada barra
    offset = (np.arange(n_escen) - (n_escen - 1) / 2) * ancho

    fig, ax = plt.subplots(figsize=(max(10, n_vars * 1.8), 5))
    fig.suptitle(titulo, fontweight='bold', fontsize=13)

    for k, (label, color, w, b) in enumerate(escenarios):
        valores = list(w) + [b]
        bars = ax.bar(x + offset[k], valores, width=ancho * 0.92,
                      color=color, alpha=0.85, label=label,
                      edgecolor='white', linewidth=0.5)
        # etiqueta encima/debajo de cada barra
        for bar, val in zip(bars, valores):
            ypos = bar.get_height() + (0.01 * max(abs(v) for v in valores or [1]))
            va   = 'bottom' if val >= 0 else 'top'
            ypos = bar.get_y() + bar.get_height() + (0.01 if val >= 0 else -0.01) * abs(val + 1e-9)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.002 if val >= 0 else -0.002),
                    f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=7.5, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, rotation=20, ha='right')
    ax.set_ylabel('Valor del peso')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.legend(framealpha=0.85)
    ax.grid(axis='y', **GRID_KW)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _guardar(nombre_archivo)

# ===========================================================================
# BLOQUE A — GD LEARNING RATES
# ===========================================================================

def graficas_gd_lr(datos, iteraciones=500):
    configs = [
        (0.05,   C1, 'lr = 0.05'),
        (0.005,  C2, 'lr = 0.005'),
        (0.0001, C3, 'lr = 0.0001'),
    ]
    historiales = []
    pesos_finales = []   # ← acumulamos pesos para la grafica extra
    for lr, color, label in configs:
        w, b, hist = gradient_descent(datos, lr=lr, iteraciones=iteraciones)
        historiales.append((hist, color, label))
        pesos_finales.append((label, color, w, b))

    # ── A.1, A.2, A.3: una figura individual por lr ──────────────────────
    for (lr_val, _, _), (hist, color, label) in zip(configs, historiales):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f'GD — Convergencia  ({label})',
                     fontweight='bold', fontsize=13)

        for ax, log_y in zip(axes, [False, True]):
            ax.plot(hist, color=color, **LINE_KW)
            _anotar_final(ax, hist, color, log_y)
            escala = 'Escala logarítmica' if log_y else 'Escala lineal'
            _fmt_ax(ax, escala, 'Iteración', 'MSE', log_y=log_y)

        plt.tight_layout()
        fname = ('GD_lr_' + str(lr_val)).replace('.','p') + '.png'
        _guardar(fname)

    # ── Comparacion de los 3 lr ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('GD — Comparación de Learning Rates',
                 fontweight='bold', fontsize=13)

    for hist, color, label in historiales:
        for ax, log_y in zip(axes, [False, True]):
            ax.plot(hist, color=color, label=f'{label}  (final={hist[-1]:.1f})',
                    **LINE_KW)
            _anotar_final(ax, hist, color, log_y)

    _fmt_ax(axes[0], 'Escala lineal',        'Iteración', 'MSE', log_y=False)
    _fmt_ax(axes[1], 'Escala logarítmica',   'Iteración', 'MSE', log_y=True)
    for ax in axes:
        ax.legend(framealpha=0.8)

    plt.tight_layout()
    _guardar('GD_lr_comparacion.png')

    # ── NUEVA: pesos finales por learning rate ───────────────────────────
    grafica_pesos_finales(
        pesos_finales,
        titulo='GD — Pesos finales por Learning Rate',
        nombre_archivo='GD_lr_pesos_finales.png'
    )


# ===========================================================================
# BLOQUE B — GD INICIALIZACIONES
# ===========================================================================

def _calcular_w_univariados(datos):
    """Pendiente de regresion simple y~x_j para cada feature j."""
    n = len(datos)
    w_univ = []
    for j in range(len(datos[0][0])):
        xj = [x[j] for x,_ in datos]
        ys = [y     for _,y in datos]
        sx=sum(xj); sy=sum(ys)
        sxy=sum(a*b for a,b in zip(xj,ys))
        sx2=sum(a**2 for a in xj)
        d = n*sx2 - sx**2
        w_univ.append((n*sxy - sx*sy)/d if abs(d)>1e-10 else 0.0)
    b_univ = (sum(y for _,y in datos) -
              sum(w*sum(x[j] for x,_ in datos)
                  for j,w in enumerate(w_univ))) / n
    return w_univ, b_univ

def graficas_gd_init(datos, lr=0.01, iteraciones=500):
    p = len(datos[0][0])
    random.seed(0)
    w_rand = [random.uniform(-1, 1) for _ in range(p)]
    w_univ, b_univ = _calcular_w_univariados(datos)

    configs = [
        ([0.0]*p, 0.0,    C1, 'B.1  Ceros'),
        (w_rand,  0.0,    C2, 'B.2  Aleatorio'),
        (w_univ,  b_univ, C3, 'B.3  Univariados'),
    ]

    historiales = []
    pesos_finales = []   # ← acumulamos pesos para la grafica extra
    for w_i, b_i, color, label in configs:
        w, b, hist = gradient_descent(datos, lr=lr, iteraciones=iteraciones,
                                      w_init=w_i, b_init=b_i)
        historiales.append((hist, color, label))
        pesos_finales.append((label, color, w, b))

    # ── B.1, B.2, B.3: figura individual ────────────────────────────────
    for hist, color, label in historiales:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f'GD — Inicialización: {label}  (lr={lr})',
                     fontweight='bold', fontsize=13)

        for ax, log_y in zip(axes, [False, True]):
            ax.plot(hist, color=color, **LINE_KW)
            _anotar_final(ax, hist, color, log_y)
            escala = 'Escala logarítmica' if log_y else 'Escala lineal'
            _fmt_ax(ax, escala, 'Iteración', 'MSE', log_y=log_y)

        axes[0].annotate(f'MSE₀={hist[0]:.0f}',
                         xy=(0, hist[0]),
                         xytext=(len(hist)*0.05, hist[0]),
                         fontsize=8.5, color=color,
                         arrowprops=dict(arrowstyle='->', color=color,
                                         lw=1.2))

        plt.tight_layout()
        tag = label.split()[0].replace('.','')
        _guardar(f'GD_init_{tag}.png')

    # ── Comparacion de las 3 inicializaciones ────────────────────────────
    # B.1 Ceros y B.2 Aleatorio siguen trayectorias casi identicas y se
    # solapan en el plot. Para que las tres curvas sean siempre visibles
    # se asigna un estilo de linea distinto a cada una y se ordena el
    # z-order para que la azul (ceros) quede siempre al frente.
    estilos = [
        dict(linestyle='-',  linewidth=2.6, alpha=0.95, zorder=4),  # B.1 Ceros  — solida, encima
        dict(linestyle='--', linewidth=2.2, alpha=0.90, zorder=3),  # B.2 Aleatorio — discontinua
        dict(linestyle='-',  linewidth=2.2, alpha=0.92, zorder=2),  # B.3 Univariados — solida
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('GD — Comparación de Inicializaciones de Pesos  (lr=0.01)',
                 fontweight='bold', fontsize=13)

    for (hist, color, label), estilo in zip(historiales, estilos):
        mse0 = hist[0]
        for ax, log_y in zip(axes, [False, True]):
            ax.plot(hist, color=color,
                    label=f'{label}  (MSE₀={mse0:.0f}, final={hist[-1]:.1f})',
                    **estilo)
            _anotar_final(ax, hist, color, log_y)

    _fmt_ax(axes[0], 'Escala lineal\n(B.1 sólida, B.2 discontinua)',
            'Iteración', 'MSE', log_y=False)
    _fmt_ax(axes[1], 'Escala logarítmica\n(B.1 sólida, B.2 discontinua)',
            'Iteración', 'MSE', log_y=True)
    for ax in axes:
        ax.legend(framealpha=0.8, loc='upper right')

    mse_min = min(hist[-1] for hist,_,_ in historiales)
    for ax in axes:
        ax.axhline(mse_min, color='gray', linestyle=':', linewidth=1.2,
                   label=f'MSE mínimo={mse_min:.1f}')

    plt.tight_layout()
    _guardar('GD_init_comparacion.png')

    # ── NUEVA: pesos finales por inicializacion ──────────────────────────
    grafica_pesos_finales(
        pesos_finales,
        titulo='GD — Pesos finales por Inicialización de Pesos  (lr=0.01)',
        nombre_archivo='GD_init_pesos_finales.png'
    )


# ===========================================================================
# BLOQUE C — SGD LEARNING RATES
# ===========================================================================

def graficas_sgd_lr(datos, epocas=100):
    configs = [
        (0.001, C1, 'lr = 0.001'),
        (0.01,  C2, 'lr = 0.01'),
        (0.05,  C3, 'lr = 0.05'),
    ]
    historiales = []
    for lr, color, label in configs:
        _, _, hist = stochastic_gradient_descent(datos, lr=lr, epocas=epocas)
        historiales.append((hist, color, label))

    # ── C.1, C.2, C.3: figura individual ────────────────────────────────
    for (lr_val, _, _), (hist, color, label) in zip(configs, historiales):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f'SGD — Convergencia  ({label})',
                     fontweight='bold', fontsize=13)

        for ax, log_y in zip(axes, [False, True]):
            ax.plot(hist, color=color, **LINE_KW)
            _anotar_final(ax, hist, color, log_y)
            escala = 'Escala logarítmica' if log_y else 'Escala lineal'
            _fmt_ax(ax, escala, 'Época', 'MSE', log_y=log_y)

        plt.tight_layout()
        fname = ('SGD_lr_' + str(lr_val)).replace('.','p') + '.png'
        _guardar(fname)

    # ── Comparacion de los 3 lr ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('SGD — Comparación de Learning Rates',
                 fontweight='bold', fontsize=13)

    for hist, color, label in historiales:
        for ax, log_y in zip(axes, [False, True]):
            ax.plot(hist, color=color,
                    label=f'{label}  (final={hist[-1]:.1f})', **LINE_KW)
            _anotar_final(ax, hist, color, log_y)

    _fmt_ax(axes[0], 'Escala lineal',      'Época', 'MSE', log_y=False)
    _fmt_ax(axes[1], 'Escala logarítmica', 'Época', 'MSE', log_y=True)
    for ax in axes:
        ax.legend(framealpha=0.8)

    plt.tight_layout()
    _guardar('SGD_lr_comparacion.png')


# ===========================================================================
# CORRELACIONES
# ===========================================================================

def _pearson(a, b):
    """Coeficiente de correlacion de Pearson entre dos listas."""
    n  = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
    da  = math.sqrt(sum((ai - ma) ** 2 for ai in a))
    db  = math.sqrt(sum((bi - mb) ** 2 for bi in b))
    return num / (da * db) if da * db > 1e-10 else 0.0

def grafica_correlaciones(datos_raw):
    """
    Dos graficas usando datos SIN normalizar (datos_raw):

    1. Barras horizontales: correlacion de Pearson de cada feature con y.
       Barras rojas = correlacion positiva, azules = negativa.

    2. Heatmap de la matriz de correlacion completa (features + y).
       Escala divergente rojo-azul, valor r dentro de cada celda.
    """
    import numpy as np

    n_feat  = len(datos_raw[0][0])
    nombres = FEATURE_NAMES[:n_feat] + ['nota_final ( Y )']
    n_vars  = len(nombres)

    # ── extraer columnas ──────────────────────────────────────────────────
    cols = [[] for _ in range(n_vars)]
    for x, y in datos_raw:
        for j, xj in enumerate(x):
            cols[j].append(xj)
        cols[n_feat].append(y)

    # ── calcular matriz de correlacion completa ───────────────────────────
    matriz = [[_pearson(cols[i], cols[j]) for j in range(n_vars)]
              for i in range(n_vars)]
    r_con_y = [matriz[j][n_feat] for j in range(n_feat)]

    # ── GRAFICA 1: barras — correlacion feature vs y ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle('Correlación de Pearson: features vs consumo_kwh',
                 fontweight='bold', fontsize=13)

    colores = ['#E53935' if r >= 0 else '#2196F3' for r in r_con_y]
    bars    = ax.barh(list(range(n_feat)), r_con_y, color=colores,
                      alpha=0.85, edgecolor='white', height=0.55)

    for bar, r in zip(bars, r_con_y):
        xoff = 0.015 if r >= 0 else -0.015
        ha   = 'left'  if r >= 0 else 'right'
        ax.text(r + xoff, bar.get_y() + bar.get_height() / 2,
                f'{r:+.3f}', va='center', ha=ha, fontsize=9,
                fontweight='bold',
                color='#E53935' if r >= 0 else '#2196F3')

    ax.set_yticks(list(range(n_feat)))
    ax.set_yticklabels(FEATURE_NAMES[:n_feat])
    ax.set_xlabel('Coeficiente de Pearson  r')
    ax.set_xlim(-1.15, 1.15)
    ax.axvline(0,    color='gray', linewidth=0.9)
    ax.axvline( 0.5, color='gray', linewidth=0.6, linestyle=':')
    ax.axvline(-0.5, color='gray', linewidth=0.6, linestyle=':')
    ax.axvline( 0.8, color='gray', linewidth=0.6, linestyle='--')
    ax.axvline(-0.8, color='gray', linewidth=0.6, linestyle='--')
    ax.text(0.51, -0.85, '|r|>0.5\ncorr. moderada', fontsize=7, color='gray', ha='left')
    ax.text(0.81, -0.85, '|r|>0.8\ncorr. fuerte',   fontsize=7, color='gray', ha='left')
    ax.grid(axis='x', **GRID_KW)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _guardar('correlacion_features_vs_y.png')

    # ── GRAFICA 2: heatmap de la matriz completa ──────────────────────────
    mat_np = np.array(matriz)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle('Matriz de correlación de Pearson\n(features + consumo_kwh)',
                 fontweight='bold', fontsize=13)

    im = ax.imshow(mat_np, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='r')

    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(nombres, rotation=35, ha='right', fontsize=8.5)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(nombres, fontsize=8.5)

    for i in range(n_vars):
        for j in range(n_vars):
            r = matriz[i][j]
            color_txt = 'white' if abs(r) > 0.65 else 'black'
            ax.text(j, i, f'{r:.2f}', ha='center', va='center',
                    fontsize=8, color=color_txt, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    _guardar('correlacion_heatmap.png')


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':

    datos_raw = cargar_csv('caso2_notas.csv')
    medias, stds = calcular_media_std(datos_raw)
    datos = normalizar_zscore(datos_raw, medias, stds)
    print(f'Dataset: {len(datos)} muestras, {len(datos[0][0])} features\n')

    ITER   = 1000
    EPOCAS = 100

    print('=== Correlaciones ===')
    grafica_correlaciones(datos_raw)

    print('\n=== GD: Learning Rates ===')
    graficas_gd_lr(datos, iteraciones=ITER)

    print('\n=== GD: Inicializaciones ===')
    graficas_gd_init(datos, lr=0.01, iteraciones=ITER)

    print('\n=== SGD: Learning Rates ===')
    graficas_sgd_lr(datos, epocas=EPOCAS)

    print(f'\nTodas las figuras guardadas en: {OUTPUT_DIR}/')
    archivos = [
        'correlacion_features_vs_y.png', 'correlacion_heatmap.png',
        'GD_lr_0p05.png', 'GD_lr_0p005.png', 'GD_lr_0p0001.png',
        'GD_lr_comparacion.png', 'GD_lr_pesos_finales.png',
        'GD_init_B1.png', 'GD_init_B2.png', 'GD_init_B3.png',
        'GD_init_comparacion.png', 'GD_init_pesos_finales.png',
        'SGD_lr_0p001.png', 'SGD_lr_0p01.png', 'SGD_lr_0p05.png',
        'SGD_lr_comparacion.png',
    ]
    for f in archivos:
        print(f'  outputs/{f}')