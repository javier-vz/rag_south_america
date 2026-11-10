# RAG_SA — Español / Português (v6)

Esta versión **solo modifica la presentación y el soporte bilingüe** de la v5: conserva el estilo oscuro/verde, el grafo RDF, la búsqueda BM25, MiniLM opcional y las respuestas generadas con Groq. No escribe en GitHub.

## Cambios de esta entrega

- Selector **Español / Português** que traduce las pestañas, botones, filtros, tarjetas, mensajes y sección «Acerca de / Sobre o projeto».
- Preguntas sugeridas en ambos idiomas, incluyendo **«¿Qué sabes sobre la lengua kariri?» / «O que você sabe sobre a língua kariri?»**.
- Groq redacta en el idioma elegido, incluso si se escribe una pregunta en el otro idioma. Las relaciones y nombres RDF originales no se traducen ni se modifican.
- Fichas del explorador con resúmenes amables construidos a partir de relaciones del grafo, sin exhibir los resúmenes fuente en inglés.
- «Acerca de» ampliado con explicación amigable del proyecto y sección técnica; su inspiración se describe como **manuscrito de investigación**, no como artículo publicado. El manuscrito entregado se titula *Graph-Based Retrieval-Augmented Generation for South American Indigenous Languages: Toward Equitable Access to Digital Cultural Heritage* y enumera a Javier Vera Zúñiga, Dina Cornejo Meza y Wenceslao Palma Muñoz.
- Pequeñas ampliaciones de vocabulario en portugués para interpretar preguntas sobre países, familias, cantidades y hablantes, sin cambiar el algoritmo de recuperación.

## Inicio

1. Copia los archivos de datos originales del proyecto o ejecuta `python descargar_datos.py` (solo lectura).
2. Instala las dependencias: `pip install -r requirements.txt`.
3. Copia `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml` y configura **`GROQ_API_KEY`** y, si hace falta, **`GROQ_MODEL`**.
4. Ejecuta `streamlit run app_lenguas.py`.
5. Cambia el idioma con el selector **🌐 Idioma / Idioma** situado debajo del encabezado.

No incluyas `.streamlit/secrets.toml` en GitHub. En Streamlit Community Cloud, configura los secretos desde la administración de la aplicación.

## Archivos de datos

`grafo_enriquecido.ttl` es obligatorio. `embed_matrix_v2.npy` e `id_map_v2.pkl` son opcionales y habilitan la búsqueda semántica ampliada. No se modifican ni se incluyen en el ZIP.

## Pruebas

```bash
python -m pytest -q
```

Las pruebas usan un grafo pequeño de ejemplo, sin necesitar una API key real. La generación en Groq debe comprobarse en tu despliegue. El historial ya creado no se retraduce al cambiar de idioma; las nuevas respuestas aparecen en el idioma elegido.
