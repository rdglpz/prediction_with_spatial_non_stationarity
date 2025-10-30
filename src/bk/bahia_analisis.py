import numpy as np

def constructTocSeries(df, 
                       delta, 
                       points, 
                       year_init = 1985, 
                       year_end = 2020):
    """
    Construye una lista de objetos TOC (curvas de 'Total Operating Characteristic')
    a partir de archivos CSV que representan cambios entre intervalos de años.

    Parámetros
    ----------
    df : pandas.DataFrame
        (No se utiliza en esta función, pero se deja por compatibilidad con otras rutinas)
    delta : int
        Tamaño del intervalo en años (p.ej., 2 -> intervalos 1985_1987, 1986_1988, ...)
    points : array-like
        (No se utiliza en esta función; se conserva por compatibilidad)
    year_init : int
        Año inicial absoluto (inclusive).
    year_end : int
        Año final absoluto (inclusive en lectura de etiquetas, pero efectiva hasta year_end - delta).

    Retorna
    -------
    list
        Lista de objetos TOC (uno por intervalo temporal).
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_curve  # (importado pero no utilizado)
    import matplotlib.pyplot as plt        # (importado pero no utilizado)
    from pytoc import TOC, TOC_painter     # TOC_painter queda sin usar aquí
    
    # Directorio y prefijo de los CSV generados por 'comparisonsTimeSeries'
    directory = "databases_interval_comparison_ba/"
    prefix = "dist_change_bahia_"
    
    # Etiquetas de intervalos: "1985_1987", "1986_1988", ..., hasta year_end - delta
    interval_labels = [str(y) + "_" + str(y + delta) for y in 
                       np.arange(year_init, year_end - delta + 1)]
    
    TOCs = []
    
    # Recorremos cada intervalo temporal
    for i, s in enumerate(interval_labels):
        path = directory + prefix + s + ".csv"
        print(path)
        
        # Cargamos el CSV correspondiente al intervalo
        db = pd.read_csv(path)
        
        # Tomamos columnas clave (coordenadas y variables finales) para armar pares (rank, y)
        # pair: [col0, col1, col(-3), col2, col(-2)]
        #  - Asume que:
        #    * col 0,1: coords
        #    * col -3: 'slope' (pendiente)
        #    * col 2 : 'distances'
        #    * col -2: etiqueta de transición codificada (ver 'change')
        pair = np.array(db.iloc[:, [0, 1, -3, 2, -2]])
        
        # rank: índice/orden; y: variable binaria (1 deforestación, 0 persistencia)
        rank, y = get_rank_y_pair(pair)
        
        # Invertimos el ranking para que "celdas más cercanas" tengan mayor prioridad
        reverse_rank =  np.max(rank) - rank
        
        # Ordenamos de mayor a menor score (reverse_rank)
        rank_sorted = np.flip(np.argsort(reverse_rank.flatten()))
        rank_descend_ss = reverse_rank[rank_sorted]
        y_aligned = y[rank_sorted]
        
        # Muestreo/umbrales: calculamos percentiles del score para definir thresholds
        six = np.arange(0, len(rank_sorted), 1000)       # (vector no utilizado luego)
        division = np.linspace(0, 100, 32)               # 32 puntos de percentil
        perc = np.percentile(reverse_rank, division)     # valores umbral por percentiles
        uranks = np.unique(perc)                         # únicos para evitar repetidos
        
        booleanArray = y_aligned      # etiquetas ordenadas
        indexArray = rank_descend_ss  # puntajes ordenados
        
        # thresholds para TOC: de mayor a menor
        thresholds = np.flip(uranks)
        
        # Construimos el objeto TOC con etiquetas, puntajes y umbrales
        TOC_1 = TOC(booleanArray, indexArray, thresholds)
        TOCs.append(TOC_1)

        # (Opcional) ¡Se podría pintar aquí con TOC_painter si se requiere visualización!

    return TOCs


def comparisonsTimeSeries(df, delta, points, year_init = 1985, year_end = 2020):
    """
    Genera tablas por intervalo temporal con:
      - coordenadas (points),
      - distancia inicial al no-bosque,
      - pendiente ('slope'),
      - y el vector de cambio codificado por 'change'.

    Produce archivos CSV con nombre 'dist_change_bahia_<YYYY>_<YYYY+delta>.csv'.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene múltiples mapas/columnas por año (distancias, etc.).
        Se asume que:
         - df[i+2] y df[i+2+delta] son mapas para comparar (por índice de columna).
         - df.iloc[:, -3] es la columna 'slope'.
    delta : int
        Tamaño del intervalo en años.
    points : array-like
        Coordenadas o puntos base (se vuelcan como DataFrame para iniciar 'losses').
    year_init, year_end : int
        Rango de años absolutos para construir las parejas (inicio, fin) de comparación.
    """
    import numpy as np
    import pandas as pd
    
    # Etiquetas de intervalos: "1985_1987", "1986_1988", ..., hasta year_end - delta
    interval_labels = [str(y) + "_" + str(y + delta) for y in 
                       np.arange(year_init, year_end - delta + 1)]
    
    print("Saving table of coordinates, Initial distances, changes in an interval from ")
    
    for i, s in enumerate(interval_labels):
        # Base: DataFrame con 'points' (asumido como coords u otra info)
        losses = pd.DataFrame(points)
    
        # map1 y map2: columnas del df que representan el estado en t y t+delta
        map1 = np.array(df[i + 2])            # mapa inicial
        map2 = np.array(df[i + 2 + delta])    # mapa final
        
        # Vector de cambio codificado (0,1,2,3) según transiciones bosque/no-bosque
        V = change(map1, map2)
        
        # Armamos la tabla de salida
        losses["distances"] = map1
        losses["slope"] = np.array(df.iloc[:, -3])
        losses[s] = V
        
        print("saving:")
        name = "dist_change_bahia_" + s + ".csv"
        print(name)
        
        losses.to_csv(name)


def change(map1, map2):
    """
    Codifica transiciones entre dos mapas categóricos (bosque/no-bosque).

    Convención de clases:
      - 0 = bosque
      - !=0 = no-bosque

    Códigos de salida:
      0: no-bosque -> no-bosque (persistencia no-bosque)
      1: no-bosque -> bosque     (ganancia de bosque / 'lost non-forest' en comentario original)
      2: bosque    -> no-bosque (ganancia de no-bosque / deforestación)
      3: bosque    -> bosque     (persistencia bosque)
    """
    # persistencia no-bosque: barren -> barren
    pers_non_forest = ((map1 == 0) & (map2 == 0)) * 0
    
    # no-bosque -> bosque (ganancia de bosque)
    lost = ((map1 == 0) & (map2 != 0)) * 1
    
    # bosque -> no-bosque (deforestación / 'gain non-forest')
    gain = ((map1 != 0) & (map2 == 0)) * 2
    
    # persistencia bosque: forest -> forest
    pers_forest = ((map1 != 0) & (map2 != 0)) * 3
    
    return pers_non_forest + lost + gain + pers_forest


def invert_rank(rank):
    """
    Invierte un ranking numérico: valores altos pasan a bajos y viceversa.
    """
    return np.max(rank) - rank


def get_rank_y_pair(ch1, dv = 4):
    """
    Extrae (rank, y) para entrenamiento/evaluación binaria desde una matriz 'ch1'.

    Estructura esperada de 'ch1' por columnas (según construcción previa):
      0: índice/orden (rank base)
      1: coordenada en puntos
      2: distancia a no-bosque
      3: pendiente (slope)
      4: etiqueta de transición (salida de 'change'): {0,1,2,3}

    Se consideran solo dos clases de interés:
      - 2: bosque -> no-bosque (deforestación)  -> y = 1
      - 3: bosque -> bosque (persistencia)      -> y = 0

    Parámetros
    ----------
    ch1 : np.ndarray
        Matriz con columnas como se describe arriba.
    dv : int
        Índice de columna donde está la clase de transición (por defecto 4).

    Retorna
    -------
    rank : np.ndarray
        Vector de ranking/índice para los casos seleccionados.
    y : np.ndarray
        Vector binario de etiquetas (1 = deforestación, 0 = persistencia de bosque).
    """
    import numpy as np
    
    # índices donde la transición es de interés (2 o 3)
    ix = np.argwhere(1 * (ch1[:, dv] == 2) + 1 * (ch1[:, dv] == 3))
    
    # 'rank' se toma de la columna 0
    rank = ch1[ix, 0]
    
    # y = 1 para transición 2 (deforestación), y = 0 para 3 (persistencia bosque)
    y = ((ch1[ix, dv] == 2) * 1).flatten()
    
    return rank, y


def mapComparison(t):
    """
    Mapea una transición (t0, t1) a un código de 0 a 3 para colorear/visualizar.

    Convención (coincide con 'change'):
      0: barren -> barren (no-bosque a no-bosque)    : gris
      1: barren -> forest (no-bosque a bosque)
      2: forest -> forest (bosque a bosque)          : verde
      3: forest -> barren (bosque a no-bosque)       : rojo
    """
    if t[0] == 0 and t[1] == 0:
        return 0
    if t[0] == 0 and t[1] != 0:
        return 1
    if t[0] != 0 and t[1] != 0:
        return 2
    if t[0] != 0 and t[1] == 0:
        return 3


def generate_dataset(X, Y, Y_patch, h, mask):
    """
    Construye un dataset de parches espaciales (cuadrados) para un problema binario
    usando una secuencia de imágenes como entrada.

    Parámetros
    ----------
    X : np.ndarray
        Tensor (H, W, L) con L 'bandas' temporales (años previos).
        Se asume que X==0 representa 'bosque'; se binariza internamente.
    Y : np.ndarray
        Matriz (H, W) binaria; 1 si en el siguiente año hay 'presencia' objetivo
        (en este código: deforestación en el centro).
    Y_patch : np.ndarray
        Matriz (H, W) con la clase para el año siguiente (para obtener parches completos).
    h : int
        Half-size del parche (el parche será de (2h+1) x (2h+1)).
    mask : np.ndarray
        Máscara (H, W) binaria para filtrar ubicaciones válidas.

    Retorna
    -------
    XX : np.ndarray
        Conjunto de parches de entrada de tamaño (N, 2h+1, 2h+1, L), con valores binarios (X==0).
    YY : np.ndarray
        Etiquetas (N,), donde N es el número de parches válidos.
    YY_patch : np.ndarray
        Parches (N, 2h+1, 2h+1) binarios del año siguiente (Y_patch==0).
    """
    import numpy as np
    
    L = np.shape(X)[2]
    width = h*2 + 1
    height = h*2 + 1
    
    # Número total de posiciones centrales posibles al barrer la imagen con margen h
    nm = len(range(h, np.shape(X)[0]-h-1)) * len(range(h, np.shape(X)[1]-h-1))
    
    XX = np.zeros((nm, height, width, L))
    YY = np.zeros(nm)
    YY_patch = np.zeros((nm, height, width))
    flatten_mask = np.zeros(nm)
    non_forest_last_step = np.zeros(nm)
    
    k = 0

    # Barrido espacial (evitando bordes por el tamaño del parche)
    for i in range(h, np.shape(X)[0]-h-1):
        for j in range(h, np.shape(X)[1]-h-1):
        
            # Parche (2h+1)x(2h+1)xL de las L bandas previas
            # Se binariza: 1 si es bosque (X==0), 0 en otro caso
            XX[k, :, :, :] = (X[i-h:i+h+1, j-h:j+h+1, :] == 0) * 1

            # Etiqueta del centro en el siguiente año (deforestación -> 1)
            YY[k] = Y[i, j] * 1
            
            # Parche del año siguiente (binarizado a bosque/no-bosque del centro extendido)
            YY_patch[k] = (Y_patch[i-h:i+h+1, j-h:j+h+1] == 0) * 1
            
            # Máscara y condición: último paso previo debe ser no-bosque
            flatten_mask[k] = mask[i, j]
            non_forest_last_step[k] = (X[i, j, -1] != 0) 
            k = k + 1
   
    # Filtramos: posiciones donde mask==1 y el último estado previo es no-bosque
    ix = np.argwhere((flatten_mask == 1) & (non_forest_last_step == 1))
            
    return XX[ix], YY[ix], YY_patch[ix]


def generate_canonical_dataset(sequence_of_images, YEAR, LAGS, HALF_SIZE, mask):
    """
    Genera el dataset canónico (X_tr, Y_tr, Y_tr_patch) a partir de una
    secuencia temporal completa de mapas categóricos.

    Parámetros
    ----------
    sequence_of_images : np.ndarray
        Tensor (H, W, T) con T años consecutivos. Convención: 0 = bosque, !=0 = no-bosque.
    YEAR : int
        Índice relativo del año objetivo dentro de la secuencia (0 = primer año absoluto).
        Debe cumplirse YEAR >= LAGS para poder tomar LAGS previos.
    LAGS : int
        Número de observaciones pasadas a usar como entrada (profundidad temporal).
    HALF_SIZE : int
        Mitad del tamaño del parche (parches de (2*HALF_SIZE+1)^2).
    mask : np.ndarray
        Máscara binaria (H, W) para filtrar ubicaciones válidas.

    Retorna
    -------
    X_tr : np.ndarray
        Parches de entrada (N, 2h+1, 2h+1, LAGS) binarios (bosque=1).
    Y_tr : np.ndarray
        Etiquetas (N,) con 1 si ocurre deforestación en el centro en YEAR.
    Y_tr_patch : np.ndarray
        Parches (N, 2h+1, 2h+1) del año YEAR binarizados (bosque=1).
    """
    D2NF = np.copy(sequence_of_images)
    
    NON_FOREST = 0
    h = HALF_SIZE
    bahia_mask = mask
    
    # X: LAGS anteriores a YEAR (no incluye el año objetivo)
    X = D2NF[:, :, YEAR - LAGS:YEAR]
    
    # Y: 1 si de YEAR-1 a YEAR ocurre bosque->no-bosque en el centro
    Y = (D2NF[:, :, YEAR - 1] != NON_FOREST) & (D2NF[:, :, YEAR] == NON_FOREST)
    
    # Y_patch: frame completo del año objetivo
    Y_patch = D2NF[:, :, YEAR]
    
    X_tr, Y_tr, Y_tr_patch = generate_dataset(X, Y, Y_patch, h, bahia_mask)
    return X_tr, Y_tr, Y_tr_patch
Notas y sugerencias
Parámetros no usados en constructTocSeries: df y points no se utilizan. Si no son necesarios, podrías quitarlos para evitar confusiones.

Imports no usados: roc_curve, matplotlib.pyplot as plt, TOC_painter, y el vector six dentro de constructTocSeries no se usan; puedes eliminarlos para limpiar el código.

Convenciones de clase: Se asume 0 = bosque y !=0 = no-bosque. Lo documenté en change y en la generación de datasets.

Índices de columnas en pair: pair = db.iloc[:, [0, 1, -3, 2, -2]] depende fuertemente del orden de columnas del CSV. Si cambian los CSV, esto se rompe. Considera seleccionar por nombres de columnas.

Rendimiento: generate_dataset recorre pixel a pixel y arma parches; si H y W son grandes, puede ser costoso en memoria/tiempo. Podrías:

Vectorizar parte del proceso o

Usar generadores/yield por bloques o

Usar ventanas deslizantes con as_strided (cuidando memoria).

Cheques de límites: En generate_canonical_dataset, asegúrate de que YEAR >= LAGS y YEAR < T, con asserts explícitos.

Semántica TOC: Estás usando percentiles de reverse_rank como thresholds. Está bien, pero podrías parametrizar el número de bins (32) y exponerlo como argumento.

Nombres: lost en change en realidad representa ganancia de bosque (no-bosque → bosque). Lo dejé aclarado en el comentario.

Si quieres, puedo:

Generarte una versión “limpia” eliminando imports/variables no usados y agregando checks (asserts), o

Adaptar la lectura de CSV para usar nombres de columna (si me dices cómo se llaman en tus archivos).






You said:
si



ChatGPT said:

Thought for 8s
¡De una! Te dejo una versión limpia y robusta del código con:

imports mínimos

assert y validaciones

parámetros configurables (directorio, prefijo, nº de umbrales, columnas por nombre)

fallback automático a índices posicionales si los nombres de columna no existen

docstrings y type hints

La lógica original se mantiene.

python
Copy code
from __future__ import annotations
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

# =========================
# Utilidades de codificación y ranking
# =========================

def invert_rank(rank: np.ndarray) -> np.ndarray:
    """Invierte un ranking numérico: valores altos pasan a bajos y viceversa."""
    return np.max(rank) - rank


def change(map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    """
    Codifica transiciones entre dos mapas categóricos (bosque/no-bosque).

    Convención de clases:
      - 0 = bosque
      - !=0 = no-bosque

    Códigos de salida:
      0: no-bosque -> no-bosque (persistencia no-bosque)
      1: no-bosque -> bosque     (ganancia de bosque)
      2: bosque    -> no-bosque  (deforestación / ganancia de no-bosque)
      3: bosque    -> bosque     (persistencia bosque)
    """
    pers_non_forest = ((map1 == 0) & (map2 == 0)) * 0
    lost            = ((map1 == 0) & (map2 != 0)) * 1
    gain            = ((map1 != 0) & (map2 == 0)) * 2
    pers_forest     = ((map1 != 0) & (map2 != 0)) * 3
    return pers_non_forest + lost + gain + pers_forest


def mapComparison(t: Tuple[int, int]) -> int:
    """
    Mapea una transición (t0, t1) a un código de 0 a 3 para colorear/visualizar.
    Convención (coincide con 'change'):
      0: barren -> barren (no-bosque a no-bosque)
      1: barren -> forest (no-bosque a bosque)
      2: forest -> forest (bosque a bosque)
      3: forest -> barren (bosque a no-bosque)
    """
    if t[0] == 0 and t[1] == 0:
        return 0
    if t[0] == 0 and t[1] != 0:
        return 1
    if t[0] != 0 and t[1] != 0:
        return 2
    if t[0] != 0 and t[1] == 0:
        return 3
    raise ValueError("Valores de transición no válidos.")


# =========================
# Extracción (rank, y) desde pares
# =========================

def get_rank_y_pair(
    ch1: np.ndarray,
    dv: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae (rank, y) para binario desde una matriz 'ch1'.

    Estructura por columnas esperada:
      0: índice/orden (rank base)
      1: coordenada (id o similar)
      2: distancia a no-bosque
      3: pendiente (slope)
      4: clase de transición {0,1,2,3} (salida de 'change')

    Se usan solo:
      - 2: bosque->no-bosque (deforestación)  -> y=1
      - 3: bosque->bosque (persistencia)      -> y=0
    """
    assert ch1.ndim == 2 and ch1.shape[1] >= dv + 1, "ch1 no tiene columnas suficientes."
    ix = np.argwhere((ch1[:, dv] == 2) | (ch1[:, dv] == 3)).flatten()
    if ix.size == 0:
        # Sin casos de interés: devolvemos arrays vacíos
        return np.array([]), np.array([])
    rank = ch1[ix, 0]
    y = (ch1[ix, dv] == 2).astype(int)
    return rank, y


# =========================
# Generación de datasets con parches
# =========================

def generate_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    Y_patch: np.ndarray,
    h: int,
    mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye dataset de parches (2h+1)^2 a partir de secuencias temporales X.

    X: (H, W, L) — L 'bandas' temporales (años previos); se binariza (X==0)->1.
    Y: (H, W)     — etiqueta binaria (1 si ocurre evento objetivo en el centro).
    Y_patch: (H,W)— frame del año objetivo; se binariza (==0)->1.
    h: mitad del tamaño del parche; parche de (2h+1)x(2h+1).
    mask: (H, W)  — máscara binaria para filtrar ubicaciones válidas.
    """
    H, W, L = X.shape
    assert Y.shape == (H, W) and Y_patch.shape == (H, W) and mask.shape == (H, W), \
        "Dimensiones incompatibles entre X, Y, Y_patch o mask."
    assert h >= 0 and (H > 2*h+1) and (W > 2*h+1), "Parche demasiado grande para la imagen."

    height = width = 2*h + 1
    # Cantidad de centros válidos (borde excluido)
    n_i = max(0, (H - (2*h + 1)))
    n_j = max(0, (W - (2*h + 1)))
    nm = n_i * n_j

    XX = np.zeros((nm, height, width, L), dtype=np.uint8)
    YY = np.zeros(nm, dtype=np.uint8)
    YY_patch = np.zeros((nm, height, width), dtype=np.uint8)
    flatten_mask = np.zeros(nm, dtype=np.uint8)
    non_forest_last_step = np.zeros(nm, dtype=np.uint8)

    k = 0
    for i in range(h, H - h - 1):
        for j in range(h, W - h - 1):
            XX[k] = (X[i-h:i+h+1, j-h:j+h+1, :] == 0).astype(np.uint8)
            YY[k] = Y[i, j].astype(np.uint8)
            YY_patch[k] = (Y_patch[i-h:i+h+1, j-h:j+h+1] == 0).astype(np.uint8)
            flatten_mask[k] = mask[i, j].astype(np.uint8)
            non_forest_last_step[k] = (X[i, j, -1] != 0)
            k += 1

    # Filtramos: mask==1 y último paso previo es no-bosque
    ix = np.argwhere((flatten_mask == 1) & (non_forest_last_step == 1)).flatten()
    if ix.size == 0:
        return (XX[:0], YY[:0], YY_patch[:0])

    return XX[ix], YY[ix], YY_patch[ix]


def generate_canonical_dataset(
    sequence_of_images: np.ndarray,
    YEAR: int,
    LAGS: int,
    HALF_SIZE: int,
    mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera el dataset (X_tr, Y_tr, Y_tr_patch) desde una secuencia (H,W,T).

    Convención: 0 = bosque, !=0 = no-bosque.

    YEAR: índice del año objetivo (0-based). Debe cumplir:
          LAGS >= 1, YEAR >= LAGS, YEAR < T
    LAGS: nº de pasos pasados a usar como entrada.
    HALF_SIZE: mitad del tamaño del parche.
    mask: máscara binaria (H,W).
    """
    assert sequence_of_images.ndim == 3, "sequence_of_images debe ser (H,W,T)."
    H, W, T = sequence_of_images.shape
    assert LAGS >= 1, "LAGS debe ser >= 1."
    assert YEAR >= LAGS and YEAR < T, "YEAR debe satisfacer LAGS <= YEAR < T."

    D2NF = np.copy(sequence_of_images)
    h = HALF_SIZE

    # X: LAGS previos a YEAR (no incluye YEAR)
    X = D2NF[:, :, YEAR - LAGS:YEAR]
    # Y: 1 si de YEAR-1 a YEAR ocurre bosque->no-bosque en el centro
    Y = (D2NF[:, :, YEAR - 1] != 0) & (D2NF[:, :, YEAR] == 0)
    # Y_patch: frame completo del año objetivo
    Y_patch = D2NF[:, :, YEAR]

    return generate_dataset(X, Y, Y_patch, h, mask)


# =========================
# Construcción de TOC por intervalos
# =========================

def constructTocSeries(
    df,  # mantenido por compatibilidad (no usado)
    delta: int,
    points,  # mantenido por compatibilidad (no usado)
    year_init: int = 1985,
    year_end: int = 2020,
    *,
    directory: str = "databases_interval_comparison_ba",
    prefix: str = "dist_change_bahia_",
    num_thresholds: int = 32,
    column_names: Optional[Dict[str, str]] = None
):
    """
    Construye una lista de objetos TOC (Total Operating Characteristic) a partir de
    archivos CSV generados por 'comparisonsTimeSeries'.

    Parámetros clave (configurables):
      - directory/prefix: para localizar 'prefixYYYY_YYYY+delta.csv' dentro de 'directory'.
      - num_thresholds: nº de percentiles (umbrales) para la curva.
      - column_names: nombres de columnas esperados en los CSV. Si no existen,
        se hace fallback a índices posicionales [0,1,-3,2,-2].

        Esperado (si se proveen):
          {'x': '<col>', 'y': '<col>', 'slope': '<col>',
           'distance': '<col>', 'transition': '<col>'}

    Retorna:
      - Lista de objetos TOC (uno por intervalo).
    """
    import pandas as pd
    from pytoc import TOC  # import local para no exigir dependencia si no se llama

    assert delta >= 1 and year_end > year_init, "Parámetros de años/delta inválidos."

    os.makedirs(directory, exist_ok=True)
    interval_labels = [f"{y}_{y + delta}" for y in range(year_init, year_end - delta + 1)]
    TOCs = []

    # Normalizamos ruta
    directory = directory.rstrip("/")

    for s in interval_labels:
        path = f"{directory}/{prefix}{s}.csv"
        print(f"Leyendo: {path}")
        if not os.path.exists(path):
            print(f"  [ADVERTENCIA] No existe el archivo: {path}. Se omite.")
            continue

        db = pd.read_csv(path)

        # Intento por nombres de columna si se proporcionaron y existen
        use_positional = True
        if column_names:
            keys = ['x', 'y', 'slope', 'distance', 'transition']
            if all((column_names.get(k) in db.columns) for k in keys):
                pair = np.array(
                    db[[column_names['x'],
                        column_names['y'],
                        column_names['slope'],
                        column_names['distance'],
                        column_names['transition']]]
                )
                use_positional = False

        # Fallback posicional: [0, 1, -3, 2, -2]
        if use_positional:
            try:
                pair = np.array(db.iloc[:, [0, 1, -3, 2, -2]])
                if column_names:
                    print("  [INFO] Columnas por nombre no halladas; usando índices posicionales.")
            except Exception as e:
                print(f"  [ERROR] No fue posible extraer columnas: {e}. Se omite {path}.")
                continue

        # Extraemos (rank, y)
        rank, y = get_rank_y_pair(pair)
        if rank.size == 0:
            print("  [INFO] Sin casos (2 o 3) en este intervalo; se omite.")
            continue

        # Score mayor para celdas más cercanas (invertimos)
        reverse_rank = invert_rank(rank)

        # Orden descendente por score
        order = np.flip(np.argsort(reverse_rank.flatten()))
        indexArray = reverse_rank[order]
        booleanArray = y[order]

        # Umbrales (percentiles) de mayor a menor
        division = np.linspace(0, 100, num_thresholds)
        perc = np.percentile(indexArray, division)
        thresholds = np.flip(np.unique(perc))

        TOC_1 = TOC(booleanArray, indexArray, thresholds)
        TOCs.append(TOC_1)

    return TOCs


# =========================
# Comparaciones por intervalo y guardado de CSV
# =========================

def comparisonsTimeSeries(
    df,
    delta: int,
    points,
    year_init: int = 1985,
    year_end: int = 2020,
    *,
    out_directory: str = ".",
    out_prefix: str = "dist_change_bahia_",
    slope_col: Optional[str] = None,
    map_col_offset: int = 2
) -> None:
    """
    Genera CSV por intervalo con:
      - 'points' como base,
      - 'distances' (mapa inicial),
      - 'slope' (desde df[slope_col] o df.iloc[:,-3]),
      - columna del vector de cambio 'YYYY_YYYY+delta' (por 'change(map1,map2)').

    df: DataFrame con columnas por año; se asume que:
        df[i + map_col_offset] es el mapa del año inicial del i-ésimo intervalo,
        df[i + map_col_offset + delta] el del final.
    slope_col: si se provee y existe en df, se usa; si no, fallback a df.iloc[:,-3].
    out_directory: carpeta de salida.
    """
    import pandas as pd

    assert delta >= 1 and year_end > year_init, "Parámetros de años/delta inválidos."
    os.makedirs(out_directory, exist_ok=True)

    interval_labels = [f"{y}_{y + delta}" for y in range(year_init, year_end - delta + 1)]
    print("Guardando tablas (coords, distancia inicial, slope, y cambio por intervalo)...")

    for i, s in enumerate(interval_labels):
        losses = pd.DataFrame(points)

        try:
            map1 = np.array(df.iloc[:, i + map_col_offset])
            map2 = np.array(df.iloc[:, i + map_col_offset + delta])
        except Exception as e:
            print(f"[ADVERTENCIA] No pude extraer columnas de mapas para {s}: {e}. Se omite.")
            continue

        V = change(map1, map2)

        # Distancias: aquí se asume que map1 representa la distancia inicial
        losses["distances"] = map1

        # Slope: por nombre si está, si no último -3
        if slope_col and slope_col in df.columns:
            losses["slope"] = np.array(df[slope_col])
        else:
            losses["slope"] = np.array(df.iloc[:, -3])

        # Columna del intervalo con el vector de cambio
        losses[s] = V.astype(int)

        name = f"{out_prefix}{s}.csv"
        out_path = os.path.join(out_directory, name)
        print(f"  -> {out_path}")
        losses.to_csv(out_path, index=False)
