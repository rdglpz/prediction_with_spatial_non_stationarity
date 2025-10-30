.
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   │   └── gtiff
│   │       ├── bahia_slope_gtiff
│   │       ├── loss_dist
│   │       │   ├── loss1985_dist_geotiff
│   │       │   ├── .
│   │       │   ├── .
│   │       │   └── loss2020_dist_geotiff
│   │       ├── mask_bahia_gtiff
│   │       └── recent_deforestation_density
│   │           ├── deforestation1985_1986_dens_geotiff
│   │           ├── .
│   │           ├── .
│   │           ├── .
│   │           └── deforestation2019_2020_dens_geotiff
│   └── raw
│       └── brazil
│           └── Brazil01.zip<--Base de datos original
├── models
├── notebooks
│   └── GLP.ipynb <-- Libreta principal con los experimentos
├── references
├── reports
├── requirements.txt
├── setup.py
├── src
│   └── deforestation_utils.py <-Archivo comentado mas importante con las funciones necesarias para transofrmar los datos geoespaciales a secuencia de histogramas a ser modelados
├── test_environment.py
└── tox.ini