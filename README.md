# RAG South America — v6.9

**v6.8: continuidad de la conversación · v6.9: navegación desde las respuestas**

Esta actualización parte de la versión trilingüe v6.7. Mantiene el diseño oscuro/verde, las tres lenguas de la interfaz, la recuperación RDF/BM25 (con MiniLM opcional) y **Groq como generador de todas las respuestas**.

## Qué se añadió

### v6.8 — Seguimiento de preguntas

Después de «¿Qué sabes sobre la lengua Kariri?», puedes preguntar «¿Y dónde se hablaba?» o «¿Se está intentando recuperar?» sin repetir Kariri. Funciona también con *E onde era falada?* y *And where was it spoken?*.

- Se conserva **el identificador RDF de la lengua** que acaba de consultarse, no una transcripción completa que pueda introducir errores.
- Una pregunta sobre otra lengua explícitamente nombrada cambia de tema.
- Las consultas generales sobre un país no heredan accidentalmente la lengua anterior.
- Si el nombre de una lengua no aparece en el grafo, el asistente puede reconocer esa falta de información sin asignarle los datos de otra lengua.
- Limpiar la conversación elimina también la memoria del tema.
- Las preguntas siguen pasando por Groq y conservan el idioma elegido en la interfaz.

La continuidad se centra en una **lengua identificable**. No es una memoria general ni resuelve todas las referencias ambiguas.

### v6.9 — Navegación contextual

Debajo de las respuestas aparecen, cuando las relaciones RDF lo permiten, botones como:

- **Ver ficha: Jaqaru** → abre la ficha en «Explorar».
- **Explorar: Aymaran** → abre la agrupación y permite elegir una de sus lenguas registradas.
- **Ver país: Perú** → abre «Explorar» con el país seleccionado y sus agrupaciones filtradas.

Se mantienen las pestañas originales mediante el estado controlado de `st.tabs`, sin reemplazar la interfaz por otra navegación. Los botones **no inventan países ni familias**. Por ejemplo, Kariri puede tener descripción histórica sobre Brasil, pero si el nodo no posee la relación `wikidata_countries`, no se crea un botón de país desde ese dato textual.

## Instalación / actualización

**Si ya tienes la v6.7:** sustituye los archivos de la actualización mínima:

1. `app_lenguas.py`
2. `rag_core.py`
3. `locales.py`
4. `requirements.txt`
5. Añade el nuevo `conversation_nav.py`.

No reemplaces tus datos RDF ni tus secretos. El módulo `explore_tree.py` y `styles.py` no han cambiado.

**Requisito importante:** usamos `st.tabs(..., on_change="rerun", key="main_tabs")`, disponible en **Streamlit 1.55 o superior**. Por eso se actualizó el mínimo de `requirements.txt` a `streamlit>=1.55,<2`.

```bash
pip install -r requirements.txt
streamlit run app_lenguas.py
```

Si comienzas desde cero, usa `python descargar_datos.py` o copia los tres archivos originales del repositorio: `grafo_enriquecido.ttl`, `embed_matrix_v2.npy` e `id_map_v2.pkl`. La búsqueda ampliada con embeddings sigue siendo opcional.

## Pruebas

```bash
python -m pytest -q
```

**95 pruebas automatizadas aprobadas** en el entorno de preparación, incluyendo regresiones de la versión anterior, preguntas de seguimiento en los tres idiomas y navegación por relaciones RDF. La conexión con la cuenta real de Groq y el comportamiento visual en tu despliegue de Streamlit requieren una comprobación manual.

### Comprobación manual recomendada

1. Pregunta en español «¿Qué sabes sobre la lengua kariri?»; luego «¿Y dónde se hablaba?». La segunda respuesta debe seguir refiriéndose a Kariri.
2. Pregunta por Jaqaru; utiliza «Ver ficha», «Explorar: Aymaran» y «Ver país: Perú». Deben abrir «Explorar» con el contexto indicado.
3. Cambia a Português e English; formula preguntas de seguimiento y verifica que la respuesta use el idioma elegido.
4. Pregunta por una lengua nueva y confirma que deja de usar la anterior.
5. Consulta lenguas de Perú y, después, una pregunta sin referente; no debe atribuirle arbitrariamente información de Kariri.

## Archivos modificados

`app_lenguas.py`, `rag_core.py`, `locales.py`, `requirements.txt`.

**Archivo nuevo:** `conversation_nav.py` y pruebas `tests/test_conversation_navigation_v69.py`.

**Sin cambios:** `grafo_enriquecido.ttl`, `explore_tree.py`, `styles.py`, embeddings, motor BM25, secretos y modelo Groq configurado.

**No se modificó GitHub.**
