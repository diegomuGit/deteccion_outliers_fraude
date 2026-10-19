# Detección de outliers en fraude con tarjetas de crédito

Taller de detección de anomalías (MIAX) sobre datos de transacciones con tarjeta de crédito.

## Estructura

```
data/        datos (zips con los CSV; los CSV descomprimidos se ignoran en git)
notebooks/   notebooks del taller y de modelos predictivos
outputs/     CSV de scores de producción generado por el notebook
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

Los notebooks del taller y de modelos de construcción leen los ZIP directamente desde `data/`, sin descomprimirlos.

## Modelos predictivos con construcción

[`notebooks/modelos_predictivos_construccion.ipynb`](notebooks/modelos_predictivos_construccion.ipynb)
adapta la estructura de `credit-card-fraud-detection-predictive-models.ipynb` al dataset
`data/credit_card_construction.csv.zip`, con las variables `X1`–`X20`, `Amount` y `Class`.
Incluye exploración con Matplotlib y compara Random Forest, AdaBoost, CatBoost, XGBoost
y LightGBM. Las dependencias están incluidas en `requirements.txt`; la ejecución usa CPU
y hasta cuatro hilos por modelo.

Ejecuta sus celdas en orden desde la raíz del repositorio o desde `notebooks/`.
La división aproximada 60/20/20 mantiene las transacciones idénticas en la misma
partición. El ganador se elige por Average Precision de validación y se evalúa una
sola vez en test después de reentrenarlo con entrenamiento y validación.
El notebook incluye salidas guardadas. Su sección final clona el Random Forest elegido,
lo entrena con todo construcción y puntúa `data/credit_card_production.csv.zip`,
usando los mismos predictores y conservando el orden original de las transacciones.
El modelo para el envío se mantiene separado del utilizado en el test interno.

El archivo listo para enviar al profesor es
`outputs/scores_produccion_random_forest.csv`: una columna de probabilidades de fraude,
sin cabecera ni índice. Se verifica al volver a leerlo: 113.923 filas, valores finitos
entre 0 y 1 y los mismos scores en el mismo orden. Este número sigue el fichero real
de producción; el enunciado menciona 113.223 transacciones.
El AP del test interno corresponde a construcción; el AP de producción requiere
las etiquetas que no tenemos. Los CSV generados quedan excluidos de Git por `.gitignore`.
