"""Carga y uso del clasificador de ODS fuera del notebook.

Este modulo existe por una razon concreta: joblib guarda una REFERENCIA a la
clase NormalizadorTexto, no su codigo. Como el modelo se entreno dentro del
notebook, el pickle apunta a "__main__.NormalizadorTexto", y cualquier proceso
que lo cargue debe tener esa clase disponible bajo ese nombre.

    normalizar_token y NormalizadorTexto son una COPIA de la celda de §2.3 del
    notebook, que es la fuente. Si se modifican alli hay que replicar el cambio
    aqui y volver a generar el .joblib, o el modelo dejara de reconstruirse.

Diferencia deliberada frente al notebook: alli las stopwords se materializan al
definir la clase (para resolver un problema de serializacion con cloudpickle en
la busqueda de hiperparametros); aqui se cargan de forma perezosa dentro de fit.
El motivo es que la aplicacion nunca llama a fit -el modelo ya viene entrenado,
con sus stopwords y su stemmer dentro- y asi importar este modulo no depende del
corpus de NLTK. El comportamiento de transform, que es lo unico que se usa en
inferencia, es identico.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

AQUI = Path(__file__).resolve().parent
RUTA_MODELO = AQUI.parent / "modelos" / "clasificador_ods.joblib"
RUTA_NLTK = AQUI.parent / "nltk_data"


def normalizar_token(texto):
    """Pasa un texto a minusculas y le quita los acentos, conservando la ene.

    NFKD descompone la 'n' en 'n' mas una tilde combinante, asi que filtrar las
    combinantes la convertiria en una 'n' normal. Se sustituye por un caracter
    de control antes de descomponer y se restaura despues.
    """
    texto = str(texto).lower().replace("ñ", "\x01")
    texto = "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )
    return texto.replace("\x01", "ñ")


def _cargar_stopwords(idioma):
    """Lee las stopwords de NLTK. Solo hace falta si se reentrena."""
    import nltk

    if str(RUTA_NLTK) not in nltk.data.path:
        nltk.data.path.insert(0, str(RUTA_NLTK))

    from nltk.corpus import stopwords

    try:
        palabras = stopwords.words(idioma)
    except LookupError as error:
        raise RuntimeError(
            f"No se encontro el corpus de stopwords para {idioma!r} en "
            f"{RUTA_NLTK}. La aplicacion no lo necesita (el modelo entrenado ya "
            f"las lleva dentro); solo hace falta para reentrenar."
        ) from error

    return frozenset(normalizar_token(p) for p in palabras)


class NormalizadorTexto(BaseEstimator, TransformerMixin):
    """Normaliza textos en espanol antes de la vectorizacion.

    Aplica, en orden: minusculas, eliminacion de acentos, eliminacion de
    digitos y puntuacion, filtrado de stopwords, stemming y descarte de tokens
    demasiado cortos.
    """

    def __init__(self, aplicar_stemming=True, longitud_minima=3, idioma="spanish"):
        self.aplicar_stemming = aplicar_stemming
        self.longitud_minima = longitud_minima
        self.idioma = idioma

    def fit(self, X, y=None):
        # La aplicacion nunca llega aqui, pero el metodo debe existir:
        # check_is_fitted comprueba hasattr(estimador, "fit") y sin el
        # cada transform fallaria con "is not an estimator instance".
        from nltk.stem import SnowballStemmer

        self.stopwords_ = _cargar_stopwords(self.idioma)
        self.stemmer_ = SnowballStemmer(self.idioma)
        self.ajustado_ = True
        return self

    def _normalizar(self, texto):
        """Normaliza un unico texto y devuelve la cadena de tokens resultante."""
        texto = normalizar_token(texto)
        texto = re.sub(r"[^a-zñ\s]", " ", texto)

        tokens = [
            t for t in texto.split()
            if t not in self.stopwords_ and len(t) >= self.longitud_minima
        ]

        if self.aplicar_stemming:
            tokens = [self.stemmer_.stem(t) for t in tokens]

        return " ".join(tokens)

    def transform(self, X):
        """Aplica la normalizacion a una serie o lista de textos."""
        check_is_fitted(self)
        return [self._normalizar(t) for t in X]


# Nombres oficiales de los ODS. El modelo solo distingue 16: el corpus no cubre
# el ODS 17, asi que nunca se predecira. Se conserva la entrada por completitud,
# pero la interfaz debe recorrer modelo.classes_, no este diccionario.
NOMBRES_ODS = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educacion de calidad",
    5: "Igualdad de genero",
    6: "Agua limpia y saneamiento",
    7: "Energia asequible y no contaminante",
    8: "Trabajo decente y crecimiento economico",
    9: "Industria, innovacion e infraestructura",
    10: "Reduccion de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Produccion y consumo responsables",
    13: "Accion por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones solidas",
    17: "Alianzas para lograr los objetivos",
}

# Margen: diferencia entre la puntuacion del ganador y la del segundo. Sirve
# como senal de "el modelo duda", pero conviene no pedirle mas de lo que da.
# Medido sobre los cinco textos de §8: los aciertos sacan 0,21, 1,35 y 0,49, y
# el error 0,75. Es decir, el margen NO separa aciertos de errores en registro
# ciudadano -es la misma conclusion a la que llega §8.5-, asi que el umbral se
# fija bajo y se usa para avisar, nunca para rechazar. Los casos realmente
# vacios (margen ~0,03) los ataja antes la comprobacion de vocabulario.
MARGEN_MINIMO = 0.3


def cargar_modelo(ruta=None):
    """Devuelve el pipeline entrenado, listo para recibir texto crudo.

    Registra NormalizadorTexto en __main__ porque es ahi donde el pickle la
    busca: el modelo se serializo desde el notebook. Se hace justo antes de
    cargar y no al importar, porque Streamlit crea un modulo __main__ nuevo en
    cada ejecucion del script.
    """
    import sys

    import joblib

    ruta = Path(ruta) if ruta is not None else RUTA_MODELO
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontro el modelo en {ruta}")

    def _registrar():
        principal = sys.modules.get("__main__")
        if principal is not None:
            principal.NormalizadorTexto = NormalizadorTexto
            principal.normalizar_token = normalizar_token

    _registrar()
    try:
        return joblib.load(ruta)
    except AttributeError:
        # Streamlit pudo reemplazar __main__ entre el registro y la carga.
        _registrar()
        return joblib.load(ruta)


def a_porcentajes(puntuaciones):
    """Reescala puntuaciones a valores positivos que suman 1, via softmax.

    NO son probabilidades calibradas: un SVM maximiza el margen entre clases, no
    modela probabilidades. Sirven para comparar los candidatos entre si y para
    leer el resultado, no para afirmar "hay un 70 % de que sea el ODS 4".
    """
    # Restar el maximo de cada fila antes de exponenciar evita el desbordamiento
    # numerico. No altera el resultado: softmax es invariante a sumar una
    # constante.
    z = puntuaciones - puntuaciones.max(axis=1, keepdims=True)
    exponenciales = np.exp(z)
    return exponenciales / exponenciales.sum(axis=1, keepdims=True)


def puntuar(modelo, texto):
    """Puntua un texto, den o no probabilidades sus clasificadores."""
    if hasattr(modelo, "predict_proba"):
        return modelo.classes_, modelo.predict_proba([texto])[0], True
    return modelo.classes_, modelo.decision_function([texto])[0], False


def tiene_vocabulario(modelo, texto):
    """True si el texto conserva al menos un termino conocido por el modelo.

    No basta con mirar si el texto normalizado queda vacio: "hola" produce el
    token "hol", que no esta en el vocabulario. Lo que decide es si el vector
    que llega al clasificador tiene algun valor distinto de cero; cuando es todo
    ceros, la decision la toma unicamente el intercepto y el modelo devuelve
    siempre la misma clase con aire de certeza.
    """
    X = modelo[:-1].transform([texto])
    n = X.nnz if hasattr(X, "nnz") else int(np.count_nonzero(X))
    return n > 0


def clasificar_texto(modelo, texto, n_top=3):
    """Asigna un ODS a un texto libre y devuelve los n_top candidatos.

    Retorna un dict con 'ods', 'nombre', 'confianza', 'candidatos', 'margen',
    'calibrado' y 'sin_vocabulario'. Si el texto no aporta ningun termino
    conocido, 'sin_vocabulario' es True y no debe mostrarse la prediccion.
    """
    texto = (texto or "").strip()

    if not texto or not tiene_vocabulario(modelo, texto):
        return {
            "texto": texto,
            "sin_vocabulario": True,
            "ods": None,
            "nombre": None,
            "confianza": 0.0,
            "candidatos": [],
            "margen": 0.0,
            "calibrado": False,
            "dudoso": True,
        }

    clases, puntuaciones, es_probabilidad = puntuar(modelo, texto)
    valores = puntuaciones if es_probabilidad else a_porcentajes(puntuaciones[None, :])[0]

    orden = np.argsort(puntuaciones)[::-1]
    margen = float(puntuaciones[orden[0]] - puntuaciones[orden[1]])

    candidatos = [
        {
            "ods": int(clases[i]),
            "nombre": NOMBRES_ODS.get(int(clases[i]), f"ODS {int(clases[i])}"),
            "confianza": float(valores[i]),
            "puntuacion": float(puntuaciones[i]),
        }
        for i in orden[:n_top]
    ]

    return {
        "texto": texto,
        "sin_vocabulario": False,
        "ods": candidatos[0]["ods"],
        "nombre": candidatos[0]["nombre"],
        "confianza": candidatos[0]["confianza"],
        "candidatos": candidatos,
        "margen": margen,
        "calibrado": es_probabilidad,
        "dudoso": margen < MARGEN_MINIMO,
    }
