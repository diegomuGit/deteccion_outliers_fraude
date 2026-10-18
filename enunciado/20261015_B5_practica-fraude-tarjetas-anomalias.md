# Práctica de Detección de Anomalías — Fraude en Tarjetas de Crédito — 15 de octubre de 2026

## Índice
1. [Introducción a la práctica](#introducción-a-la-práctica)
2. [Descripción del dataset](#descripción-del-dataset)
3. [Requisitos de entrega y métrica de evaluación](#requisitos-de-entrega-y-métrica-de-evaluación)
4. [Visualización y exploración inicial de los datos](#visualización-y-exploración-inicial-de-los-datos)
5. [Preprocesamiento y tratamiento de escalas](#preprocesamiento-y-tratamiento-de-escalas)
6. [Enfoques metodológicos y selección de técnicas](#enfoques-metodológicos-y-selección-de-técnicas)
7. [Ejemplo detallado con mezcla de gaussianas](#ejemplo-detallado-con-mezcla-de-gaussianas)
8. [Consideraciones sobre el carácter no supervisado y la evolución del fraude](#consideraciones-sobre-el-carácter-no-supervisado-y-la-evolución-del-fraude)

---

## Introducción a la práctica

La práctica consiste en aplicar las técnicas de detección de anomalías vistas en el fin de semana anterior a un problema real y típico: la detección de fraude en transacciones con tarjetas de crédito. El objetivo es construir un modelo que asigne a cada transacción un *score* de anomalía, de forma que las transacciones con mayor *score* sean las más sospechosas de ser fraudulentas.

Se proporcionan dos conjuntos de datos. El de construcción contiene transacciones etiquetadas (clase 0 = legítima, clase 1 = fraude) y debe utilizarse para desarrollar y validar el modelo. El de producción contiene las mismas variables pero sin etiqueta y es el conjunto sobre el que hay que generar las predicciones finales. La práctica es abierta: se puede emplear cualquier método de los vistos (Isolation Forest, métodos basados en vecinos, autoencoders, modelos estadísticos, One-Class SVM, etc.) o variantes adicionales que se consideren adecuadas. Antes de la entrega definitiva es posible enviar predicciones intermedias por correo para recibir feedback sobre la métrica obtenida.

> **Síntesis:** La práctica plantea un problema real de detección de fraude con datos etiquetados de construcción y datos de producción sin etiqueta. El entregable principal es un *score* continuo de anomalía por transacción, y se admite cualquier técnica de detección de anomalías.

---

## Descripción del dataset

Cada fila representa una transacción real de tarjeta de crédito, previamente anonimizada. Las variables $V1, V2, \dots, V28$ (o el número exacto que aparezca) son el resultado de aplicar PCA a las variables originales. El único atributo que no ha sido anonimizado es `Amount`, el importe de la transacción (en dólares). La variable `Class` toma el valor 0 cuando la transacción es legítima y 1 cuando es fraudulenta.

El desequilibrio es extremo. En el conjunto de construcción la proporción de fraude ronda el 0,17 % (aproximadamente 1 de cada 600-700 transacciones). Este desequilibrio es coherente con la realidad del fraude con tarjetas en muchos países y obliga a utilizar métricas que no se vean dominadas por la clase mayoritaria.

No existen valores nulos y no procede una limpieza clásica de outliers: precisamente los valores extremos pueden corresponder a fraude. Las variables procedentes de PCA ya incorporan la estandarización previa de las variables originales, por lo que no es necesario (ni recomendable) estandarizarlas de nuevo de forma individual.

> **Síntesis:** El dataset contiene transacciones reales anonimizadas mediante PCA, con el importe como única variable interpretable. El fraude representa alrededor del 0,17 % de los casos y no hay valores faltantes que requieran tratamiento.

---

## Requisitos de entrega y métrica de evaluación

La entrega consiste en dos elementos:

1. El notebook con todo el desarrollo realizado.
2. Un único fichero CSV que contenga exactamente una columna de *scores* (una puntuación por fila) para las transacciones del conjunto de producción. El fichero debe tener tantas filas como transacciones hay en producción (113 223 en el momento de la explicación).

El *score* debe ser una variable continua. Cuanto más alto sea el valor, mayor debe ser la evidencia de que la transacción es anómala (fraude). No se exige una probabilidad calibrada ni una etiqueta binaria; el *score* puede ser cualquier magnitud monótona con la “rareza” del caso (por ejemplo, el negativo del logaritmo de la densidad, el error de reconstrucción de un autoencoder, el *score* de Isolation Forest, etc.).

La métrica utilizada para evaluar las entregas es el área bajo la curva de precisión-recall (AUPR), implementada en scikit-learn como `average_precision_score`. Es imprescindible indicar que la clase positiva es la 1 (fraude). Esta métrica depende únicamente del ranking inducido por los *scores*: multiplicar todos los *scores* por una constante positiva no altera el resultado. Por tanto, la escala absoluta del *score* es irrelevante siempre que el orden relativo sea correcto.

La fecha límite de entrega es el jueves 29 de octubre a las 18:30. Se recomienda validar el modelo internamente mediante una partición estratificada del conjunto de construcción en entrenamiento y test antes de generar las predicciones de producción.

> **Síntesis:** Hay que entregar el notebook y un CSV de *scores* continuos para producción. La métrica oficial es el AUPR (average precision) con clase positiva = 1; solo importa el ranking de los *scores*.

---

## Visualización y exploración inicial de los datos

Antes de entrenar cualquier modelo es útil visualizar la estructura de los datos. Dado que las variables $V1$ y $V2$ son las dos primeras componentes principales, un *scatter plot* de $V1$ frente a $V2$ coloreado por la clase ofrece una primera impresión de la separabilidad. Se recomienda utilizar Plotly Express por su interactividad.

Con un número elevado de puntos (del orden de 100 000) conviene muestrear (por ejemplo, 10 000 puntos) para que la visualización sea fluida. Ajustar la opacidad de los marcadores ayuda a percibir la densidad de la clase mayoritaria. Mediante el argumento `hover_data` se pueden mostrar variables adicionales (en particular `Amount`) al pasar el ratón por encima de un punto, lo que permite explorar de forma rápida si el importe guarda alguna relación visible con el fraude.

Estas visualizaciones revelan que $V1$ tiene una escala y una varianza mucho mayores que $V2$, algo esperado: PCA ordena las componentes por varianza explicada decreciente.

> **Síntesis:** Un *scatter* interactivo de las dos primeras componentes principales, con color por clase, opacidad y datos al pasar el ratón, constituye un buen primer análisis exploratorio y muestra la diferencia de escalas entre componentes.

---

## Preprocesamiento y tratamiento de escalas

No se requiere limpieza de valores faltantes ni eliminación de outliers. La decisión crítica es el tratamiento de la escala de las variables, que depende del algoritmo elegido.

- Isolation Forest y la mezcla de gaussianas (Gaussian Mixture) no se ven afectados por la escala de las variables.
- Los autoencoders, las redes neuronales en general y los One-Class SVM sí son sensibles a la escala.

Cuando sea necesario normalizar, no se debe estandarizar cada variable por separado. Hacerlo igualaría la varianza de todas las componentes principales y destruiría la información que PCA ya ha codificado: la primera componente (la que resume más información de las variables originales) debe conservar una varianza mayor que las siguientes. Una alternativa sencilla y consistente consiste en dividir todas las variables $V$ por una misma constante (por ejemplo, 400, próxima a la desviación típica de $V1$). De este modo se reducen los valores a un rango manejable para redes neuronales sin alterar las relaciones relativas de importancia entre componentes.

`Amount` puede tratarse de forma independiente si el método lo requiere, pero en la mayoría de los enfoques se incluye junto con las componentes principales.

> **Síntesis:** Solo se debe escalar cuando el algoritmo lo exija. En ese caso se recomienda un escalado global (dividir todas las componentes por el mismo factor) para no anular la ordenación por varianza que proporciona PCA.

---

## Enfoques metodológicos y selección de técnicas

El problema puede abordarse tanto de forma supervisada como no supervisada. Dado el fuerte desequilibrio y la posibilidad de que en producción aparezcan patrones de fraude no vistos durante el entrenamiento, los métodos no supervisados (o semi-supervisados que aprenden únicamente el comportamiento normal) resultan especialmente adecuados.

Se pueden utilizar, entre otros:

- Isolation Forest.
- Métodos basados en densidad o vecinos (Local Outlier Factor, etc.).
- Autoencoders (el error de reconstrucción actúa como *score* de anomalía).
- Modelos estadísticos de densidad, en particular mezclas de gaussianas.
- One-Class SVM (con la precaución del escalado).

En un contexto no supervisado todas las variables tienen, a priori, la misma importancia. Por ello es fundamental el criterio de negocio: solo deben incluirse aquellas variables que se consideren realmente relevantes para el problema. Variables irrelevantes introducen ruido que el modelo no puede ignorar de forma automática. En el presente dataset las componentes principales ya constituyen una representación comprimida y anonimizada de las variables originales, por lo que el conjunto proporcionado es el que debe emplearse.

Cuando se dispone de etiquetas (conjunto de construcción) es recomendable realizar una partición estratificada en entrenamiento y test para poder estimar el AUPR antes de generar las predicciones de producción. También es posible entrenar exclusivamente con los casos de clase 0 (transacciones legítimas) y utilizar los casos de clase 1 únicamente para fijar umbrales o para evaluar, estrategia especialmente útil cuando en un sistema real se dispone de muy pocos (o ningún) ejemplo de fraude.

> **Síntesis:** Cualquier técnica de detección de anomalías es válida. Los enfoques que aprenden solo el comportamiento normal son especialmente potentes. La selección de variables debe guiarse por el conocimiento del dominio y la validación interna debe hacerse con una partición estratificada.

---

## Ejemplo detallado con mezcla de gaussianas

Un procedimiento sencillo y eficaz consiste en ajustar un modelo de mezcla de gaussianas únicamente sobre las transacciones legítimas (clase 0) del conjunto de entrenamiento.

1. Se realiza una partición estratificada del conjunto de construcción.
2. Se entrena un `GaussianMixture` (por ejemplo con 5 componentes) exclusivamente sobre los registros de clase 0.
3. Se obtiene el *score* de cada punto mediante `score_samples`. Este método devuelve el logaritmo de la densidad bajo el modelo ajustado; valores más bajos (más negativos) corresponden a puntos menos probables según la distribución de los casos normales y, por tanto, más anómalos.
4. Se ordenan las transacciones por *score* creciente. En la práctica se observa que los primeros puestos del ranking (los de menor densidad) corresponden mayoritariamente a fraudes, tanto en entrenamiento como en test.

El número de componentes es un hiperparámetro que debe ajustarse. La mezcla de gaussianas no requiere que las variables estén normalizadas. El mismo principio (entrenar solo con casos normales y usar una medida de “rareza” como *score*) se aplica de forma análoga a los autoencoders, donde el error de reconstrucción desempeña el papel de *score* de anomalía.

Este tipo de enfoque es especialmente interesante en ciberseguridad o en detección de fraude cuando se dispone de pocos o ningún ejemplo de ataque/fraude previo, o cuando los patrones de ataque evolucionan con el tiempo.

> **Síntesis:** Ajustar una mezcla de gaussianas solo sobre los casos legítimos y utilizar el logaritmo de la densidad (`score_samples`) como *score* de anomalía produce rankings muy efectivos. El método no necesita escalado y el número de componentes es el hiperparámetro principal.

---

## Consideraciones sobre el carácter no supervisado y la evolución del fraude

Aunque el conjunto de construcción dispone de etiquetas, el problema es en el fondo no supervisado: en producción pueden aparecer modalidades de fraude que no estaban presentes en los datos de entrenamiento. Los sistemas de detección de anomalías que modelan únicamente el comportamiento normal son robustos frente a esta situación.

En la práctica real es crítico asegurarse de que los casos etiquetados como “normales” no contengan fraude no detectado todavía. Fraudes de importe muy pequeño (por ejemplo, cargos de 0,5 € o 1 € repetidos en miles de tarjetas) suelen pasar desapercibidos para los titulares y pueden contaminar el conjunto de entrenamiento si no se han depurado adecuadamente.

Variables categóricas de alta cardinalidad (país, rango de IP, dominio, etc.), habituales en sistemas reales de prevención de fraude, se pueden incorporar mediante embeddings aprendidos (por ejemplo, capas de embedding en un autoencoder) o mediante técnicas de codificación controlada (one-hot limitado a las categorías más frecuentes más una categoría “otros”). En el dataset de la práctica todas las variables son numéricas, por lo que este aspecto no es necesario.

La fecha límite de entrega es el jueves 29 de octubre a las 18:30. Se recomienda aprovechar la posibilidad de enviar predicciones intermedias para recibir feedback sobre el AUPR antes de la entrega definitiva.

> **Síntesis:** El carácter evolutivo del fraude y la escasez de ejemplos de anomalías hacen especialmente adecuados los métodos que aprenden solo el comportamiento normal. La calidad de las etiquetas de los casos “ok” es crítica, y en sistemas reales las variables categóricas se incorporan preferentemente mediante embeddings.
