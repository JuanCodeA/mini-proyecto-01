# Generación de paletas de colores a partir de imágenes

Mini Proyecto 1 — **Machine Learning No Supervisado**
Maestría en Inteligencia Artificial, Semestre 2

Método que, dada la imagen de una obra de arte, extrae sus colores dominantes mediante
agrupación y genera un muestrario representativo, **decidiendo por sí mismo cuántos grupos
formar y con qué algoritmo**.

## Qué hace

```
imagen → pipeline de preparación → agrupación con selección automática → paleta
                                            │
                                            └──→ proyección t-SNE de la nube de color
```

- **Prepara** la imagen con un `Pipeline` de scikit-learn: RGB, redimensionado a 400 px
  conservando la relación de aspecto, muestreo de píxeles y conversión a **CIELAB**, para que
  la distancia euclidiana corresponda a la diferencia de color percibida.
- **Compara tres algoritmos** —K-Means, Mean Shift y Gaussian Mixture— sobre 28 configuraciones
  por imagen, y elige combinando un piso de fidelidad cromática (ΔE) con el consenso de
  Silhouette, Davies-Bouldin y Calinski-Harabasz.
- **Genera el muestrario** usando el medoide de cada grupo, fusionando colores redundantes y
  ordenando por luminosidad.
- **Verifica que generaliza** aplicando el método a cuatro géneros que no participaron en
  ninguna decisión de diseño.

## Cómo ejecutarlo

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name mini-proyecto \
    --display-name "Python (Mini-Proyecto)"
```

Abre `mini-proyecto.ipynb`, selecciona el kernel **Python (Mini-Proyecto)** y ejecuta
*Restart & Run All*.

**No hacen falta credenciales de Kaggle ni conexión a Internet.** El catálogo y las 24 imágenes
que el notebook utiliza viajan en `data/`. El código conserva la descarga desde Kaggle como
respaldo, por si se cambia la semilla o se añaden géneros nuevos.

Requiere Python 3.13. Las versiones exactas con las que se ejecutó están en
`requirements.lock.txt`.

## Estructura

```
mini-proyecto.ipynb     el método completo, documentado paso a paso
mini-proyecto.html      exportación del notebook ya ejecutado
requirements.txt        dependencias
requirements.lock.txt   versiones exactas de la ejecución de referencia
data/classes.csv        catálogo WikiArt (80.043 obras)
data/imagenes/          las 24 obras usadas, en carpetas por género
```

## Datos

Las imágenes provienen del conjunto [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart)
publicado en Kaggle por *steubk*, con licencia **CC0 (dominio público)**. Solo se incluyen las
24 obras que el notebook analiza; el conjunto completo pesa 31,45 GB.
