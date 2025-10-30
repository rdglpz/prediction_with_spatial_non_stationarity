import numpy as np
import georaster


def makeGrid(rows, columns, block_size = 100):
    """
    Crea una rejilla (grid) entera en la que cada bloque de tamaño `block_size x block_size`
    recibe un ID incremental (0, 1, 2, ...). Útil para agrupar píxeles por bloques.

    Parámetros
    ----------
    rows : int
        Alto total de la grilla (número de renglones).
    columns : int
        Ancho total de la grilla (número de columnas).
    block_size : int, por defecto 100
        Tamaño (lado) de cada bloque cuadrado.

    Retorna
    -------
    grid : np.ndarray (rows, columns) de dtype int
        Matriz con IDs por bloque.
    """
    total_size_x = rows
    total_size_y = columns

    # Matriz de ceros donde escribiremos los IDs de bloque
    grid = np.zeros((total_size_x, total_size_y), dtype=int)

    # Recorremos la imagen por bloques y asignamos un ID a cada bloque
    for i in range(0, total_size_x, block_size):
        for j in range(0, total_size_y, block_size):
            # ID del bloque = índice de bloque en X * (número de bloques en Y + 1) + índice de bloque en Y
            grid[i:i+block_size, j:j+block_size] = (
                (i // block_size) * ((total_size_y // block_size) + 1) + (j // block_size)
            )

    return grid


def readMask(dir =  "../data/processed/gtiff/mask_bahia_gtiff"):
    """
    Lee una máscara geoespacial desde un GeoTIFF (vía georaster.MultiBandRaster),
    convierte a booleano y luego pone NaN donde la máscara es 0.

    Parámetros
    ----------
    dir : str
        Ruta al archivo de máscara (GeoTIFF).

    Retorna
    -------
    bahia_mask : np.ndarray (H, W) con valores {1, NaN}
        1 donde la máscara es válida, NaN donde no.
    """
    mask = georaster.MultiBandRaster(dir)

    # mask.r suele ser (H, W, C). Se considera válida la celda si r == 255 (canal 0)
    # Se invierte con !=True para obtener True donde r != 255; luego se toma el canal 0.
    bahia_mask = (mask.r == 255) != True
    bahia_mask = bahia_mask[:, :, 0]

    # Donde la máscara sea 0 -> poner NaN; donde sea 1 -> dejar 1
    bahia_mask = np.where(bahia_mask == 0, np.nan, bahia_mask)

    return bahia_mask


def loadMapinNumpy(directory = "../data/processed/gtiff/"):
    """
    Carga un cubo 3D (H, W, T) de mapas temporales de distancia a pérdida de bosque
    para los años 1985..2020, aplicando la máscara de bahía para poner NaN fuera.

    Parámetros
    ----------
    directory : str
        Directorio base donde están los GeoTIFFs.

    Retorna
    -------
    D2NF : np.ndarray (H, W, T)
        Tensor con las capas temporales por año, enmascaradas (NaN fuera de zona).
    """
    # Rutas relativas dentro de `directory` (no todas se usan aquí)
    recentdefpath = "recent_deforestation_density/deforestation1985_1986_dens_geotiff"
    loss_distpath = "/loss_dist/loss{a}_dist_geotiff"

    yinit = 1985
    yend = 2020

    # Cargamos máscara (1 válido / NaN fuera)
    bahia_mask = readMask()

    # Leemos un raster para conocer el tamaño (H, W)
    img = georaster.MultiBandRaster(directory + recentdefpath)
    sh = np.shape(img.r)  # usualmente (H, W, C)

    # shape del tensor temporal (H, W, número de años)
    tensor = (sh[0], sh[1], yend - yinit + 1)

    # Inicializamos el contenedor temporal
    D2NF = np.zeros(tensor)

    # Recorremos los años y cargamos el raster de cada año (canal 0),
    # aplicando la máscara (multiplica por 1 o NaN)
    years = np.arange(yinit, yend + 1)
    for i, y in enumerate(years):
        path_to_d2nf = "../data/processed/gtiff/loss_dist/loss{a}_dist_geotiff".format(a=y)
        M = georaster.MultiBandRaster(path_to_d2nf)
        D2NF[:, :, i] = M.r[:, :, 0] * bahia_mask

    return D2NF


# (esta función está repetida más abajo en tu script; la dejo aquí solo comentada.
#  Considera eliminar duplicados y mantener una única definición)
def makeGrid(rows, columns, block_size = 100):
    """
    (DUPLICADO) Crea una rejilla con IDs por bloque de tamaño `block_size`.
    """
    total_size_x = rows
    total_size_y = columns
    grid = np.zeros((total_size_x, total_size_y), dtype=int)
    for i in range(0, total_size_x, block_size):
        for j in range(0, total_size_y, block_size):
            grid[i:i+block_size, j:j+block_size] = (
                (i // block_size) * ((total_size_y // block_size) + 1) + (j // block_size)
            )
    return grid


def generateTrainingDataset(DF, F, predictionHorizon = 1, LAGS = 1, Psize = 1,
                            year0 = 1985, year1 = 2015):
    """
    Genera un dataset de entrenamiento (X, Y) a partir de data tabular DF con columnas por año.
    Extrae parches espaciales con padding y secuencias temporales de longitud LAGS, para predecir
    el valor central H pasos (predictionHorizon) hacia adelante.

    Parámetros
    ----------
    DF : pandas.DataFrame
        DataFrame cuyas columnas son años (year0..year1). Cada columna contiene el mapa vectorizado.
    F : int
        Número de filas (alto) del mapa 2D original al desvectorizar (reshape).
    predictionHorizon : int
        Horizonte de predicción H (años hacia adelante).
    LAGS : int
        Número de rezagos/observaciones pasadas a usar (profundidad temporal).
    Psize : int
        Mitad del tamaño del parche espacial (parche = (2*Psize+1)^2).
    year0 : int
        Año inicial incluido en DF.
    year1 : int
        Año final incluido en DF.

    Retorna
    -------
    X : list of np.ndarray
        Lista de tensores de entrada (parches) transpuestos (por diseño del autor).
    Y : list
        Lista con el valor escalar del centro del parche en el tiempo (t + H).
    """
    H = predictionHorizon

    # Número de columnas = número de años en DF
    interval_size = len(DF.columns)

    # Tomamos una columna para inferir el shape 2D del mapa (F x ?)
    GriddedDeforest = DF[year0].to_numpy()
    GD = np.reshape(GriddedDeforest, (F, -1))  # (-1) infiere columnas

    # PADDING tendrá dimensión: (años, F + 2*Psize, C + 2*Psize)
    shape = np.append([interval_size], [np.array(GD.shape) + Psize * 2])
    PADDING = np.zeros(shape)

    # Rellenamos PADDING con cada año, centrando el mapa y dejando borde de Psize
    for i, y in enumerate(range(year0, year1 + 1)):
        GriddedDeforest = DF[y].to_numpy()
        GD = np.reshape(GriddedDeforest, (F, -1))
        PADDING[i, Psize:Psize + GD.shape[0], Psize:Psize + GD.shape[1]] = np.ones(GD.shape) * GD

    # Rangos válidos:
    #  - RA: índices temporales donde hay suficientes LAGS y horizonte H
    #  - RV/RH: filas/columnas válidas dentro del área NO cubierta por padding
    number_of_years = PADDING.shape[2]  # OJO: por cómo está construido shape, aquí es el eje 2
    RA = [i for i in range(LAGS, number_of_years - H)]
    RV = [i for i in range(Psize, Psize + GD.shape[0])]
    RH = [i for i in range(Psize, Psize + GD.shape[1])]

    # Iterador de combinaciones (t, fila, columna)
    # NOTA: esto usa itertools.product; en tu código falta el import.
    # from itertools import product
    # iterations = product(RA, RV, RH)
    iterations = iter.product(RA, RV, RH)  # <-- tal cual lo tienes (ver nota abajo)

    X = list([])
    Y = list([])

    for i in iterations:
        x = i[0]  # índice temporal
        y = i[1]  # fila
        z = i[2]  # columna

        # Valor objetivo en el futuro (t + H) en el centro del parche
        Xnext = PADDING[x + H, y - Psize:y + Psize + 1, z - Psize:z + Psize + 1]

        # Extraemos la secuencia de LAGS anteriores: [x-LAGS, ..., x-1]
        XX = PADDING[x - LAGS:x, y - Psize:y + Psize + 1, z - Psize:z + Psize + 1]

        # El autor transpone XX (posible ajuste de orden de ejes para el modelo)
        X.append(XX.T)
        Y.append(Xnext[Psize, Psize])

    return X, Y
