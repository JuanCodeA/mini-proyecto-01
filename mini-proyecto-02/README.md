# Mini Proyecto 2 — Clasificación de textos según los ODS

Solución de procesamiento de lenguaje natural y machine learning que clasifica automáticamente
un texto en español según los **Objetivos de Desarrollo Sostenible** de la Agenda 2030.

## Qué hace

```
texto → normalización → TF-IDF → LSA → clasificador → ODS
                                  │
                                  └──→ tópicos latentes → interpretación frente a los ODS
```

- **Prepara** el texto con un `Pipeline` de scikit-learn: minúsculas, sin acentos ni dígitos,
  *stopwords* en español, *stemming* con `SnowballStemmer` y vectorización **TF-IDF** con
  unigramas y bigramas.
- **Descubre tópicos** con SVD truncado (LSA) sobre la matriz TF-IDF, e interpreta
  cualitativamente las componentes latentes frente a los ODS.
- **Compara tres clasificadores** —SVM lineal, regresión logística y Random Forest— con búsqueda
  de hiperparámetros y validación cruzada estratificada, usando **macro-F1** por el desbalance
  entre clases.
- **Mide qué aporta la reducción de dimensionalidad** entrenando sobre TF-IDF crudo y sobre el
  espacio LSA, en lugar de darla por buena.
- **Evalúa** sobre un conjunto de test que no participó en ninguna decisión.
- **Clasifica texto libre** (§8): una función que recibe cualquier texto y devuelve el ODS asignado
  con sus tres candidatos más probables. Los ejemplos están escritos en registro de participación
  ciudadana, no imitando al corpus de entrenamiento, de modo que la sección funciona como prueba de
  transferencia al caso de uso real.

## Cómo ejecutarlo

Este proyecto usa el entorno compartido de la raíz del repositorio:

```bash
../.venv/bin/pip install -r requirements.txt
```

Abre `mini-proyecto-02.ipynb`, selecciona el kernel **Python (Mini-Proyecto)** y ejecuta
*Restart & Run All*.

**No hace falta conexión a Internet.** El conjunto de datos y el corpus de *stopwords* de NLTK
viajan en el repositorio.

## Estructura

```
mini-proyecto-02.ipynb        el método completo, documentado paso a paso
requirements.txt              dependencias propias de este proyecto
data/Datos_textosODS.xlsx     9.656 textos en español etiquetados por ODS
nltk_data/                    corpus de stopwords, para no depender de nltk.download()
modelos/clasificador_ods.joblib  el pipeline entrenado, listo para clasificar
```

## Usar el modelo sin reejecutar el notebook

```python
import joblib
modelo = joblib.load("modelos/clasificador_ods.joblib")
modelo.predict(["En nuestra vereda el agua llega turbia y con mal olor."])
```

El proceso que lo cargue necesita tener definidas `normalizar_token` y `NormalizadorTexto`:
`joblib` guarda una referencia a la clase, no su código.

## Datos

Subconjunto del **OSDG Community Dataset (OSDG-CD)** 2023, traducido al español con DeepL y
aumentado mediante la API de ChatGPT.

Dos propiedades del conjunto que conviene conocer:

- **Contiene 16 de los 17 ODS.** Falta el ODS 17 (*Alianzas para lograr los objetivos*), que el
  OSDG-CD tampoco cubre por ser transversal y no identificable de forma fiable a partir del texto.
- **Las clases están desbalanceadas** en una proporción de 3,5× entre la mayor y la menor, lo que
  motiva el uso de macro-F1 y de partición estratificada.
