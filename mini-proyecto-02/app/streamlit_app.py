"""Interfaz web del clasificador de textos segun los ODS.

Punto de entrada de la aplicacion en Streamlit Community Cloud.
Toda la logica de inferencia vive en modelo_ods.py; aqui solo esta la interfaz.

El tema (fondo oscuro, color primario, tipografia) lo fija el
.streamlit/config.toml de esta misma carpeta: Streamlit lee el config de la
carpeta del script de entrada, y le da precedencia sobre el global y el del
directorio de trabajo. Lo que el tema no alcanza -las clases .ods-* de la
tarjeta de resultado- vive en estilos.css.
"""

import html
import re
from pathlib import Path

import streamlit as st

from modelo_ods import (
    COLORES_ODS,
    NOMBRES_ODS,
    cargar_modelo,
    clasificar_texto,
    nivel_de_confianza,
    terminos_influyentes,
)

AQUI = Path(__file__).resolve().parent

# Por debajo de esta longitud el texto aporta tan pocos terminos que el
# resultado no es informativo. No se bloquea, solo se avisa.
MINIMO_PALABRAS = 15

EJEMPLOS = {
    "Acueducto rural": (
        "En nuestra vereda el agua llega turbia y con mal olor. Llevamos meses "
        "reclamando al acueducto municipal y nadie viene a revisar la planta "
        "de tratamiento."
    ),
    "Escuela sin conexión": (
        "La escuela rural no tiene conexión a internet ni computadores, así que "
        "los estudiantes de bachillerato no pueden hacer las tareas que exige "
        "el nuevo plan de estudios."
    ),
    "Paneles solares": (
        "La comunidad instaló paneles solares en el centro de salud para "
        "garantizar energía durante los cortes y reducir el gasto en la planta "
        "diésel que usaban antes."
    ),
}

# Rotulo del chip para cada nivel de separacion. El texto habla de margen y no
# de acierto a proposito: §8.5 del notebook concluye que el margen no separa
# aciertos de errores. El color lo pone .ods-confianza.<nivel> en estilos.css,
# que sube de calma a alarma en vez de premiar el caso amplio con un verde.
ROTULO_NIVEL = {
    "fuerte": "Separación amplia",
    "media": "Separación moderada",
    "debil": "Decisión ajustada",
}

st.set_page_config(page_title="Clasificador de ODS", page_icon="◎", layout="centered")


@st.cache_resource(show_spinner="Cargando el modelo...")
def obtener_modelo():
    """Carga el pipeline una sola vez y lo comparte entre sesiones."""
    return cargar_modelo()


def obtener_estilos():
    """Lee la hoja de estilos.

    Sin cachear a proposito: son unos pocos kilobytes por reejecucion, y
    cachearla obligaba a reiniciar el proceso para ver cualquier cambio de
    estilo, porque runOnSave vigila los .py pero no los .css.
    """
    return AQUI.joinpath("estilos.css").read_text(encoding="utf-8")


def aviso(cuerpo_html):
    """Pinta un recuadro de advertencia con el lenguaje visual de la app."""
    st.markdown(
        f'<div class="ods-aviso"><span>&#9888;</span>'
        f'<p style="margin:0">{cuerpo_html}</p></div>',
        unsafe_allow_html=True,
    )


def fila(candidato, ancho, destacado):
    """Una fila del comparativo: numero, nombre, barra y porcentaje.

    Los colores los resuelve el CSS por clase; lo unico que viaja inline es el
    ancho de la barra, que es un dato.
    """
    clase = "ods-fila destacada" if destacado else "ods-fila"

    return (
        f'<div class="{clase}">'
        f'<span class="n">ODS {candidato["ods"]}</span>'
        f'<span class="t">{html.escape(candidato["nombre"])}</span>'
        f'<span class="ods-pista"><span class="ods-relleno" '
        f'style="width:{ancho:.1%}"></span></span>'
        f'<span class="ods-pct">{candidato["confianza"]:.1%}</span>'
        f'</div>'
    )


def tarjeta(resultado):
    """Dibuja la tarjeta del ODS asignado con sus tres candidatos."""
    candidatos = resultado["candidatos"]
    ganador = candidatos[0]
    nivel = nivel_de_confianza(resultado["margen"])

    # El color del ODS es identidad del objetivo, no parte de la paleta, y solo
    # se usa como NO-texto -borde de la tarjeta y punto-: 6 de los 16 no llegan
    # a 4.5:1 sobre el fondo.
    color = COLORES_ODS.get(ganador["ods"])
    fuerte = nivel == "fuerte"

    # Las barras se escalan al ganador para que la comparacion entre los tres se
    # lea: el softmax absoluto reparte entre 16 clases y ninguna pasaria del 30 %.
    # El porcentaje que se imprime al lado si es el valor real.
    tope = ganador["confianza"] or 1.0
    filas = "".join(
        fila(c, c["confianza"] / tope, i == 0) for i, c in enumerate(candidatos)
    )

    pie = ("Comparación relativa al candidato principal" if fuerte
           else "Varios objetivos compiten a un nivel parecido")

    borde_ods = f' style="border-left-color:{color}"' if color else ""
    punto_ods = f' style="background:{color}"' if color else ""

    st.markdown(
        f'<div class="ods-tarjeta"{borde_ods}>'
        f'<div class="ods-cabecera">'
        f'<div class="ods-identidad">'
        f'<span class="ods-punto"{punto_ods}></span>'
        f'<span class="ods-badge">ODS {ganador["ods"]}</span>'
        f'</div>'
        f'<span class="ods-nombre">{html.escape(ganador["nombre"])}</span>'
        f'<span class="ods-confianza {nivel}">{ROTULO_NIVEL[nivel]}</span>'
        f'</div>'
        f'<div class="ods-separador"></div>'
        f'<div class="ods-pie-barras">{pie}</div>'
        f'{filas}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if resultado["dudoso"]:
        aviso(
            f'<strong>El modelo duda.</strong> El segundo candidato queda a solo '
            f'{resultado["margen"]:.2f} puntos, así que esta asignación es poco '
            f'firme. Suele pasar cuando el texto toca varios objetivos a la vez o '
            f'cuando es muy corto. Recuerda que el modelo <strong>siempre escoge '
            f'el ODS más cercano</strong>, incluso si el texto no trata de ninguno.'
        )


def chips_terminos(terminos):
    """Los terminos que mas empujaron hacia el ODS asignado."""
    umbral = terminos[0][1] * 0.5
    chips = "".join(
        f'<span class="ods-termino{" fuerte" if peso >= umbral else ""}">'
        f'{html.escape(termino)}</span>'
        for termino, peso in terminos
    )
    st.markdown(
        '<div style="margin-top:1.5rem">'
        '<div class="ods-etiqueta">Términos que más pesaron en la decisión</div>'
        f'<div class="ods-terminos">{chips}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Pagina
# --------------------------------------------------------------------------

st.markdown(f"<style>{obtener_estilos()}</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="ods-barra-sup">'
    '<span class="ods-chip-proyecto">Mini Proyecto 2 · Maestría en IA</span>'
    '<span>TF-IDF + SVM lineal</span></div>',
    unsafe_allow_html=True,
)
st.markdown('<h1 class="ods-titulo">Clasificador de ODS</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="ods-sub">Pega un texto en español y el modelo indicará a cuál de los '
    '16 Objetivos de Desarrollo Sostenible corresponde.</p>',
    unsafe_allow_html=True,
)

try:
    modelo = obtener_modelo()
except Exception as error:  # noqa: BLE001 - se muestra al usuario, no se traga
    st.error(f"No se pudo cargar el modelo: {error}")
    st.stop()

if "texto" not in st.session_state:
    st.session_state.texto = ""

# Los botones se declaran ANTES del text_area: al pulsar uno, Streamlit reejecuta
# el script y el valor ya esta en session_state cuando nace el widget. Asignarlo
# despues de instanciarlo seria un error de Streamlit.
st.markdown('<div class="ods-etiqueta">Prueba con un ejemplo</div>', unsafe_allow_html=True)
for columna, (nombre, contenido) in zip(st.columns(len(EJEMPLOS)), EJEMPLOS.items()):
    if columna.button(nombre, use_container_width=True):
        st.session_state.texto = contenido

st.markdown('<div class="ods-etiqueta">Texto a clasificar</div>', unsafe_allow_html=True)
texto = st.text_area(
    "Texto a clasificar",
    key="texto",
    height=180,
    placeholder="Escribe o pega aquí el texto…",
    label_visibility="collapsed",
)

n_palabras = len(re.findall(r"\w+", texto))
corto = 0 < n_palabras < MINIMO_PALABRAS
st.markdown(
    f'<div class="ods-nota"><span>{n_palabras} palabras</span>'
    f'<span style="margin-left:auto">'
    f'{f"Con menos de {MINIMO_PALABRAS} palabras el resultado es poco fiable." if corto else ""}'
    f'</span></div>',
    unsafe_allow_html=True,
)

if st.button("Clasificar", type="primary", use_container_width=True,
             disabled=not texto.strip()):
    resultado = clasificar_texto(modelo, texto)

    st.markdown('<div class="ods-etiqueta" style="margin-top:2rem">RESULTADO</div>',
                unsafe_allow_html=True)

    if resultado["sin_vocabulario"]:
        aviso(
            "El texto no contiene ningún término que el modelo conozca, así que no "
            "hay nada sobre lo que decidir. Prueba con un texto más largo o más "
            "concreto."
        )
    else:
        tarjeta(resultado)

        terminos = terminos_influyentes(modelo, resultado["texto"], resultado["ods"])
        if terminos:
            chips_terminos(terminos)

        st.markdown(
            '<p class="ods-final">Los porcentajes se obtienen aplicando <em>softmax</em> '
            'a la función de decisión del SVM. <strong>No son probabilidades '
            'calibradas</strong>: sirven para comparar los candidatos entre sí, no para '
            'afirmar que hay un tanto por ciento de certeza. Las barras están escaladas '
            'al candidato principal para que la comparación se lea.</p>',
            unsafe_allow_html=True,
        )

st.divider()

with st.expander("¿Qué 16 objetivos reconoce el modelo?"):
    for numero in range(1, 17):
        st.markdown(
            f'<div class="ods-item">'
            f'<span class="ods-punto" style="background:{COLORES_ODS[numero]}"></span>'
            f'<span class="n">ODS {numero}</span>'
            f'<span class="t">{html.escape(NOMBRES_ODS[numero])}</span></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<p class="ods-final">El ODS 17 (<em>Alianzas para lograr los objetivos</em>) no '
    'aparece en el corpus de entrenamiento y nunca se predice. Tampoco existe una opción '
    '«ninguno»: si el texto no trata de ningún ODS, el modelo escogerá igualmente el más '
    'cercano. Modelo entrenado sobre el OSDG Community Dataset.</p>',
    unsafe_allow_html=True,
)
