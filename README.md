# RAG_SA · Relaciones simples + Groq (v5)

**No modifica el repositorio de GitHub.** La interfaz mantiene el diseño oscuro y verde.

## Cómo funciona

1. El grafo `grafo_enriquecido.ttl` se lee con RDFLib.
2. Se buscan relaciones directas: **lengua → país**, **lengua → familia o subgrupo** y **familia → lenguas**.
3. **Cuando la pregunta menciona una lengua concreta**, primero localiza su ficha por el nombre exacto (ignorando tildes y mayúsculas), y solo después expande relaciones inmediatas si se solicitan. Si no se menciona una lengua concreta, utiliza BM25; opcionalmente MiniLM + embeddings.
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

El error del 9 de noviembre era un `AttributeError` al leer `needs_indigenous_review` dentro de `Retriever.search`. Ahora la búsqueda usa una función independiente basada en etiquetas, sin depender de ese atributo de la clase. También se versionó la caché de Streamlit (`relaciones-v5-kariri`) para no reutilizar entidades antiguas entre despliegues.

No es necesario limpiar a mano la caché si los nuevos archivos se suben juntos. Si se mezclan ficheros de versiones distintas pueden producirse inconsistencias, por lo que **reemplaza `app_lenguas.py` y `rag_core.py` a la vez**.

## Pruebas

Ejecuta `python -m pytest -q`. Hay pruebas de relaciones RDF, búsquedas por país y familia, llamadas a Groq simuladas y regresión del `AttributeError` cuando desaparece el atributo anterior.

**Límite de verificación:** las pruebas usan un grafo pequeño de muestra. No se ha realizado una consulta a Groq con tu clave desde este entorno. El corpus original no se incluye en el ZIP para no duplicarlo.

## Corrección específica: Kariri y la familia Lengua

La pregunta «¿Qué sabes sobre la lengua Kariri?» estaba seleccionando la familia lingüística **Lengua** (`leng1262`), porque su nombre coincide con el sustantivo común «lengua». Por eso el contexto se llenaba de otras lenguas como Enlhet Norte y Enxet Sur, aunque **Kariri** (`kari1254`) sí se encontraba en el RDF.

En v5:

- Los nombres completos de **lenguas** mencionados en la pregunta tienen prioridad sobre las etiquetas de familias y la búsqueda semántica.
- La familia llamada *Lengua* se interpreta como tal solo cuando la pregunta menciona explícitamente «familia Lengua» o «subgrupo Lengua».
- Para una lengua encontrada, el contexto incluye su ficha y una descripción más completa; en el caso de Kariri, el resumen del grafo menciona su uso histórico en Brasil y la revitalización de dzubukuá.
- Si alguien pregunta por una lengua que no se identifica en el grafo, evitamos devolver candidatos arbitrarios solo porque el texto contenga la palabra «lengua».
- Groq sigue redactando todas las respuestas. **El diseño oscuro y verde no cambia.**

### Instalación de la corrección

Para actualizar desde v4 basta con sustituir `app_lenguas.py` y `rag_core.py`. Mantén `styles.py`, la carpeta `.streamlit/` con tu configuración y los archivos originales del grafo y los índices. Si prefieres una instalación limpia, el ZIP trae el proyecto completo (sin duplicar los grandes archivos del grafo).

Pruebas de regresión: `python -m pytest -q`. El caso Kariri usa los nombres y la descripción presentes en el TTL público, dentro de una pequeña fixture que reproduce la colisión con la familia Lengua. No se realizaron llamadas reales a Groq (no se dispone de la clave del usuario).
