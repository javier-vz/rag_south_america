# RAG_SA · Relaciones simples + Groq (v4)

**No modifica el repositorio de GitHub.** La interfaz mantiene el diseño oscuro y verde.

## Cómo funciona

1. El grafo `grafo_enriquecido.ttl` se lee con RDFLib.
2. Se buscan relaciones directas: **lengua → país**, **lengua → familia o subgrupo** y **familia → lenguas**.
3. Para preguntas abiertas, BM25 recupera fichas relevantes. Opcionalmente se puede activar MiniLM + embeddings, usando el índice original.
4. **Groq redacta todas las respuestas** en lenguaje natural, idealmente un párrafo amable. Si no hay datos suficientes puede decir «no lo sé». No usa cifras de hablantes a menos que las pidan. No inventa parentescos a partir de `semanticallySimilarTo`.
5. Si Groq falla, NO finge una respuesta local: muestra el mensaje de conexión y un diagnóstico en una sección desplegable.

## Instalación

1. Descomprime esta carpeta y copia `grafo_enriquecido.ttl` desde el repositorio original. También puedes ejecutar `python descargar_datos.py`, que descarga los tres archivos públicos (solo lectura).
2. Instala las dependencias: `pip install -r requirements.txt`.
3. Configura tu clave **solo** en `.streamlit/secrets.toml` (copia `.streamlit/secrets.toml.example`):

   ```toml
   GROQ_API_KEY = "gsk_..."
   GROQ_MODEL = "openai/gpt-oss-20b"
   ```

   **No subas `secrets.toml` a GitHub.** En Streamlit Community Cloud utiliza los secretos de la aplicación.
4. Ejecuta: `streamlit run app_lenguas.py`.

### Archivos de datos

- `grafo_enriquecido.ttl` es obligatorio.
- `embed_matrix_v2.npy` e `id_map_v2.pkl` son opcionales (habilitan búsqueda semántica ampliada).
- `index_faiss_v2.faiss` no se necesita.

## Problema corregido

El error del 9 de noviembre era un `AttributeError` al leer `needs_indigenous_review` dentro de `Retriever.search`. Ahora la búsqueda usa una función independiente basada en etiquetas, sin depender de ese atributo de la clase. También se versionó la caché de Streamlit (`relaciones-v4`) para no reutilizar entidades antiguas entre despliegues.

No es necesario limpiar a mano la caché si los nuevos archivos se suben juntos. Si se mezclan ficheros de v3 y v4 pueden producirse inconsistencias, por lo que **reemplaza `app_lenguas.py` y `rag_core.py` a la vez**.

## Pruebas

Ejecuta `python -m pytest -q`. Hay pruebas de relaciones RDF, búsquedas por país y familia, llamadas a Groq simuladas y regresión del `AttributeError` cuando desaparece el atributo anterior.

**Límite de verificación:** las pruebas usan un grafo pequeño de muestra. No se ha realizado una consulta a Groq con tu clave desde este entorno. El corpus original no se incluye en el ZIP para no duplicarlo.
