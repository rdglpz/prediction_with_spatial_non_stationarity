# Predicción con no estacionariedad espacial — Módulo utilitario

Este repositorio incluye utilidades para generar base de datos geoespaciales orientados a series temporales de deforestación/distancia a pérdida de bosque y a la construcción de *patches* (parches) para modelos supervisados.

## La Libreta principal está en:

```/notebooks/GLP.ipynb```

Esta incluye los experimentos incluyendo la generación de la base de datos geoespacio temporal con histogramas y ejemplos y su modelación con redes neuronales.


## Descripción general
Este módulo proporciona:
- **Herramientas de regionialización** para indexar el espacio en bloques regulares cuadrados (`makeGrid`).
- **Lectura de máscaras geoespaciales** (GeoTIFF) y aplicación como mascara binaria de presencia o número no disponible /NaN (`readMask`).
- **Generación de base de datos temporales de histograma** de tamaño BINS X años X celda.  Los histogramas indican la frecuencia de la  pérdida de bosque a cierta distancia del suelo deforestado. Desde archivos GeoTIFF (`loadMapinNumpy`).
- **Generación de datasets de parches** espaciales-temporales para entrenamiento supervisado (`generateTrainingDataset`).

---

## Requisitos
- Python ≥ 3.9
- `numpy`
- `georaster` (o paquete equivalente que provea `MultiBandRaster`)

> **Datos:** las rutas por defecto apuntan a `../data/processed/gtiff/`. Ajusta según tu organización local.


## Funciones del módulo

### `makeGrid(rows, columns, block_size=100)`
**Qué hace:** genera una rejilla 2D donde cada bloque de tamaño `block_size×block_size` recibe un **ID entero** incremental. Útil para muestreo por bloques, *tiling* o agregación espacial.

**Parámetros:**
- `rows` *(int)*: alto total (número de renglones).
- `columns` *(int)*: ancho total (número de columnas).
- `block_size` *(int, por defecto 100)*: lado del bloque cuadrado.

**Devuelve:** `np.ndarray (rows, columns)` con IDs por bloque.

---

### `readMask(dir="../data/processed/gtiff/mask_bahia_gtiff")`
**Qué hace:** lee un GeoTIFF de máscara (vía `georaster.MultiBandRaster`), toma el **canal 0** y lo convierte en matriz con **1 (válido)** y **NaN (fuera de área)**.

**Parámetros:**
- `dir` *(str)*: ruta al archivo de máscara (GeoTIFF).

**Devuelve:** `np.ndarray (H, W)` con valores en `{1, NaN}`.

> *Tip:* Verifica la convención del archivo de máscara. El código asume píxeles con valor 255 como máscara original antes de la conversión.

---

### `loadMapinNumpy(directory="../data/processed/gtiff/")`
**Qué hace:** construye un **cubo temporal** `D2NF (H×W×T)` para los años **1985..2020** leyendo GeoTIFFs anuales (`loss_dist/loss{year}_dist_geotiff`) y aplicando la máscara de bahía.

**Parámetros:**
- `directory` *(str)*: directorio base de GeoTIFFs.

**Devuelve:** `np.ndarray (H, W, T)` con los mapas anuales (canal 0) enmascarados.

> *Nota:* Internamente usa `readMask()` y espera encontrar archivos como `../data/processed/gtiff/loss_dist/loss{YEAR}_dist_geotiff`.

---

### `generateTrainingDataset(DF, F, predictionHorizon=1, LAGS=1, Psize=1, year0=1985, year1=2015)`
**Qué hace:** crea listas `X` y `Y` para **entrenamiento supervisado** a partir de una tabla `DF` con **columnas por año**. Reconstruye cada año a 2D (usando `F` filas), aplica **padding** para extraer **parches de tamaño `(2*Psize+1)`** y arma **secuencias temporales** de longitud `LAGS` para predecir el valor central a `predictionHorizon` pasos.

**Parámetros:**
- `DF` *(pandas.DataFrame)*: columnas = años (`year0..year1`), cada columna es el mapa vectorizado (flatten).
- `F` *(int)*: número de filas (alto) del mapa al hacer `reshape(F, -1)`.
- `predictionHorizon` *(int, H)*: horizontes hacia delante.
- `LAGS` *(int)*: número de rezagos (pasos pasados) a incluir.
- `Psize` *(int)*: *half-size* del parche (lado total = `2*Psize+1`).
- `year0`, `year1` *(int)*: rango de años presentes en `DF`.

**Devuelve:**
- `X` *(list[np.ndarray])*: lista de tensores con secuencias de parches transpuestos.
- `Y` *(list[float|int])*: valores objetivos (centro del parche en `t+H`).

> *Atención:* La implementación espera `itertools.product` para iterar (`product(RA, RV, RH)`), pero el código original usa `iter.product` (requiere corrección del import).

---

## Ejemplos de uso

La implementación de estas funciones y su modelización se encuentran en la carpeta 

```python
import pandas as pd
from pathlib import Path

# 1) Grilla por bloques
G = makeGrid(rows=1000, columns=1200, block_size=100)

# 2) Máscara y cubo temporal
mask = readMask("../data/processed/gtiff/mask_bahia_gtiff")
D2NF = loadMapinNumpy("../data/processed/gtiff/")  # => (H, W, T)

# 3) Dataset de entrenamiento desde DF (columnas por año)
DF = pd.DataFrame({...})
X, Y = generateTrainingDataset(DF, F=1024, predictionHorizon=1, LAGS=3, Psize=1, year0=1985, year1=2015)
```

---

## Estructura  del repositorio
```
.
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   │   └── gtiff
│   │       ├── bahia_slope_gtiff
│   │       ├── loss_dist
│   │       │   ├── loss1985_dist_geotiff
│   │       │   ├── .
│   │       │   ├── .
│   │       │   └── loss2020_dist_geotiff
│   │       ├── mask_bahia_gtiff
│   │       └── recent_deforestation_density
│   │           ├── deforestation1985_1986_dens_geotiff
│   │           ├── .
│   │           ├── .
│   │           ├── .
│   │           └── deforestation2019_2020_dens_geotiff
│   └── raw
│       └── brazil
│           └── Brazil01.zip<--Base de datos original
├── models
├── notebooks
│   └── GLP.ipynb <-- Libreta principal con los experimentos
├── references
├── reports
├── requirements.txt
├── setup.py
├── src
│   └── deforestation_utils.py <-Archivo comentado mas importante con las funciones necesarias para transofrmar los datos geoespaciales a secuencia de histogramas a ser modelados
├── test_environment.py
└── tox.ini
```

---