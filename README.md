# Detección de outliers en fraude con tarjetas de crédito

Taller de detección de anomalías (MIAX) sobre datos de transacciones con tarjeta de crédito.

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

Los datos se leen desde `data/`; descomprime los `.zip` allí si hace falta.
