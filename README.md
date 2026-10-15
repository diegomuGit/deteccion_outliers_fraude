# Detección de outliers en fraude con tarjetas de crédito

Taller de detección de anomalías (MIAX). Se entrena un modelo de mezcla gaussiana (GMM)
sobre datos normales y se usa el log-likelihood como score de anomalía, evaluándolo
sobre un conjunto de test y visualizándolo con una proyección PCA 2D.

## Estructura

```
data/        datos (zips con los CSV; los CSV descomprimidos se ignoran en git)
notebooks/   notebook con el análisis
enunciado/   PDF del taller
source/      código fuente reutilizable (vacío por ahora)
```

## Uso

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
jupyter notebook notebooks/
```

El notebook descarga/descomprime los datos en `data/` (ruta relativa `../data`).
