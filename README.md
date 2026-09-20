# Mini Proyectos — Machine Learning No Supervisado

Maestría en Inteligencia Artificial, Semestre 2.
Monorepositorio con los mini proyectos de la asignatura.

## Proyectos

| Carpeta | Proyecto | Estado |
|---|---|---|
| [`mini-proyecto-01/`](mini-proyecto-01/) | Generación de paletas de colores a partir de imágenes | Completo |
| [`mini-proyecto-02/`](mini-proyecto-02/) | Clasificación de textos según los Objetivos de Desarrollo Sostenible | Completo |

Cada carpeta es autocontenida: su notebook, sus dependencias y sus datos. El `README.md`
de cada una explica cómo ejecutarla.

El proyecto 2 va más allá del notebook: en [`mini-proyecto-02/app/`](mini-proyecto-02/app/)
hay una aplicación en Streamlit que clasifica cualquier texto con el modelo ya entrenado,
pensada para desplegarse en Streamlit Community Cloud.

## Entorno

Los dos proyectos comparten un único entorno virtual en la raíz, para no duplicar
instalaciones. Se crea una sola vez:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r mini-proyecto-01/requirements.txt
.venv/bin/pip install -r mini-proyecto-02/requirements.txt
.venv/bin/python -m ipykernel install --user --name mini-proyecto \
    --display-name "Python (Mini-Proyecto)"
```

Cada proyecto declara sus dependencias en su propio `requirements.txt` y todas se instalan
sobre el mismo entorno. La aplicación web tiene además su
`mini-proyecto-02/app/requirements.txt`, deliberadamente aparte: solo hace falta para
ejecutarla o desplegarla.

Requiere Python 3.13. Las versiones exactas de la ejecución de referencia del proyecto 1
están en `mini-proyecto-01/requirements.lock.txt`.

## Cómo ejecutar un notebook

Abre el `.ipynb` de la carpeta correspondiente y selecciona el kernel
**Python (Mini-Proyecto)**. Las rutas de datos de cada notebook son **relativas a su propia
carpeta**, así que hay que ejecutarlo desde ahí — que es lo que hacen Jupyter y VS Code por
defecto. No hace falta ninguna descarga: los datos viajan en el repositorio.

## Integrantes

- Juan Avendaño García
- Leonardo Cano Uribe
