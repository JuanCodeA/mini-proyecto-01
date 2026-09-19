"""Interfaz web del clasificador de textos segun los ODS.

Punto de entrada de la aplicacion en Streamlit Community Cloud.
Toda la logica de inferencia vive en modelo_ods.py; aqui solo esta la interfaz.
"""

import streamlit as st

from modelo_ods import MARGEN_MINIMO, cargar_modelo, clasificar_texto

st.set_page_config(page_title="Clasificador de ODS", page_icon="🎯", layout="centered")


@st.cache_resource(show_spinner="Cargando el modelo...")
def obtener_modelo():
    """Carga el pipeline una sola vez y lo comparte entre sesiones."""
    return cargar_modelo()


st.title("🎯 Clasificador de textos según los ODS")
st.write(
    "Pega un texto en español y el modelo dirá a cuál de los "
    "**Objetivos de Desarrollo Sostenible** corresponde."
)

try:
    modelo = obtener_modelo()
except Exception as error:  # noqa: BLE001 - se muestra al usuario, no se traga
    st.error(f"No se pudo cargar el modelo: {error}")
    st.stop()

texto = st.text_area(
    "Texto a clasificar",
    height=180,
    placeholder=(
        "Ejemplo: En nuestra vereda el agua llega turbia y con mal olor. "
        "Llevamos meses reclamando y nadie viene a revisar el acueducto."
    ),
)

if st.button("Clasificar", type="primary", use_container_width=True):
    resultado = clasificar_texto(modelo, texto)

    if resultado["sin_vocabulario"]:
        st.warning(
            "El texto no contiene ningún término que el modelo conozca, así que "
            "no hay nada sobre lo que decidir. Prueba con un texto más largo o "
            "más concreto."
        )
        st.stop()

    principal, *resto = resultado["candidatos"]

    st.divider()
    st.subheader(f"ODS {principal['ods']} · {principal['nombre']}")
    st.progress(min(principal["confianza"] * 2, 1.0))
    st.caption(f"Confianza relativa: {principal['confianza']:.1%}")

    if resultado["dudoso"]:
        st.warning(
            f"**El modelo duda.** El segundo candidato queda a solo "
            f"{resultado['margen']:.2f} puntos, así que esta asignación es poco "
            f"firme. Suele pasar cuando el texto toca varios objetivos a la vez "
            f"o cuando es muy corto."
        )

    if resto:
        st.markdown("**Otros candidatos**")
        for c in resto:
            izquierda, derecha = st.columns([3, 1])
            izquierda.write(f"ODS {c['ods']} · {c['nombre']}")
            derecha.write(f"{c['confianza']:.1%}")
            st.progress(min(c["confianza"] * 2, 1.0))

    st.caption(
        "Los porcentajes se obtienen aplicando *softmax* a la función de decisión "
        "del SVM. **No son probabilidades calibradas**: sirven para comparar los "
        "candidatos entre sí, no para afirmar que hay un tanto por ciento de "
        "certeza. Por eso las barras están escaladas al doble, para que se lean."
    )

st.divider()
st.caption(
    "El modelo distingue **16 objetivos**. El ODS 17 (*Alianzas para lograr los "
    "objetivos*) no aparece en el corpus de entrenamiento y nunca se predecirá. "
    "Tampoco existe una opción «ninguno»: si el texto no trata de ningún ODS, el "
    "modelo escogerá igualmente el más cercano."
)
st.caption(
    "Mini Proyecto 2 — Machine Learning No Supervisado, Maestría en Inteligencia "
    "Artificial. TF-IDF + SVM lineal sobre el OSDG Community Dataset."
)
