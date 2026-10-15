# Detección de outliers en fraude con tarjetas de crédito

Taller de detección de anomalías (MIAX). Se entrena un modelo de mezcla gaussiana (GMM)
sobre datos normales y se usa el log-likelihood como score de anomalía, evaluándolo
sobre un conjunto de test y visualizándolo con una proyección PCA 2D.

## Contenido

- `taller_detección_anomalías.ipynb`: notebook con el análisis.
- `taller_anomalías.pdf`: enunciado del taller.
- `credit_card_construction.csv.zip`, `credit_card_production.csv.zip`: datos.

## Uso

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
jupyter notebook
```

Descomprime los `.zip` en la carpeta del proyecto si el notebook espera los CSV.
