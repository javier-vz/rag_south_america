# RAG_SA — GraphRAG Explorer (versión mejorada)

**Código independiente para revisión. No modifica tu repositorio de GitHub.**

**Corrección:** Todas las preguntas —incluyendo países, familias, listas y conteos— usan el LLM de Groq cuando hay API key. El RDF calcula y verifica hechos, no redacta la respuesta final. La salida RDF directa aparece únicamente como respaldo etiquetado ante ausencia/fallo de Groq.

Mantiene el aspecto oscuro + verde de la aplicación original, pero corrige el error
404 de Groq, añade trazabilidad y mejora la recuperación semántica/léxica.

## Instalación sencilla

1. Descomprime el ZIP en una carpeta nueva, por ejemplo `RAG_SA_mejorado`.
2. Descarga los datos originales sin modificar GitHub, ejecutando en esa carpeta:

   ```bash
   python descargar_datos.py
   ```

   El script solo **lee** tres archivos públicos y comprueba sus hashes contra la versión auditada.
   Si prefieres, **copia manualmente** desde el repositorio original estos tres archivos a la raíz de esa carpeta:

   - `grafo_enriquecido.ttl`
   - `embed_matrix_v2.npy`
   - `id_map_v2.pkl`

   No necesitas `index_faiss_v2.faiss`: 961 entidades se consultan muy rápido
   con un producto matricial NumPy. Los embeddings originales se reutilizan.
3. Abre una terminal en esa carpeta e instala dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Ejecuta:

   ```bash
   streamlit run app_lenguas.py
   ```

5. Para generación **en todas las preguntas**, configura `GROQ_API_KEY` en los secretos de Streamlit,
   usando `.streamlit/secrets.toml.example` como guía. **Nunca publiques tu clave**.

   Modelo predeterminado: `openai/gpt-oss-120b`.
   Puedes cambiarlo con `GROQ_MODEL` si el proveedor actualiza su catálogo.

   **Con Groq:** todas las preguntas pasan por el LLM, incluyendo recuentos y listados.
   Los cálculos RDF proporcionan hechos verificables al LLM, pero NO sustituyen su respuesta.

   **Sin Groq:** se muestra una respuesta de respaldo claramente identificada.
   Siguen funcionando los recuentos, filtros, exploración y búsqueda. Sin `sentence-transformers` disponible
   tras instalar dependencias o sin conexión para descargar el modelo por primera
   vez, la recuperación usa BM25 de respaldo.

## Arquitectura de respuesta

```
Pregunta libre o sugerida
   ↓
Recuperación híbrida BM25 + embeddings MiniLM (RRF)
   ↓
Lectura y filtrado del grafo RDF + estadísticas estructuradas cuando aplican
   ↓
Paquete de evidencia: hechos RDF + fichas [G1], [G2]... + límites del corpus
   ↓
LLM en Groq: genera la respuesta en lenguaje natural PARA TODAS las consultas
   ↓
Respuesta con identificadores de evidencia (o respaldo etiquetado si Groq falla)
```

No hay ruta directa que omita Groq para las preguntas de país, familia o cantidades.

## ¿Qué cambia?

| Área | Original | Nuevo código |
|---|---|---|
| Error 404 | `llama-3.3-70b-versatile` | Modelo configurable y actual |
| Lectura RDF | Separación por regex | `rdflib.Graph.parse` |
| Recuperación léxica | Tokens coincidentes | BM25 |
| Recuperación híbrida | Suma de puntuaciones de escalas distintas | RRF con rankings normalizados |
| Consulta de país/familia | Ajustes ad hoc al contexto | Filtros y recuentos RDF + redacción Groq en todas las respuestas |
| Cifras de hablantes | Usadas para inferir vitalidad | Faltantes y ceros explícitos; sin afirmar extinción |
| Trazabilidad | Texto sin respaldo visual | IDs `[G1]` y enlaces a fuentes del grafo |
| Fallo del LLM | Error expuesto como respuesta | Respaldo etiquetado (no se hace pasar por respuesta de Groq) |
| Navegación | Fichas de 50 registros | Filtros + exportación CSV |
| HTML de la UI | Interpolación no escapada | `html.escape` |
| Código | Un archivo monolítico | Entrada + motor + estilos + tests |

## Limitaciones que siguen abiertas

- El grafo mezcla descripciones documentales con resúmenes enriquecidos por IA;
  todavía hay que auditar su calidad factual.
- La presencia del país en `ex:wikidata_countries` no implica cobertura completa
  ni presencia contemporánea verificada.
- No hay fechas homogéneas para cifras de hablantes. No afirmamos
  niveles de peligro lingüístico sin una taxonomía externa validada.
- Varias lenguas están asociadas a un solo país en los datos, aunque pueden
  hablarse en varios; no deducir ausencia de lenguas transfronterizas.
- Relaciones `semanticallySimilarTo` entre embeddings **no demuestran parentesco
  lingüístico**. Solo `belongsToFamily` se usa para navegar familias declaradas.
- Los cálculos de ranking y la precisión real deben medirse con un banco de
  consultas con relevancia anotada. Este proyecto incorpora pruebas técnicas,
  no una evaluación de calidad lingüística completa.
- Los recursos binarios `.npy` y `.pkl` deben venir de tu propio repositorio
  confiable: los archivos pickle no deben cargarse desde orígenes desconocidos.

## Estructura

```
RAG_SA_mejorado/
├── app_lenguas.py            # Interfaz Streamlit (conserva el estilo)
├── rag_core.py               # RDF, BM25, embeddings, filtros y respuestas
├── styles.py                 # Tema oscuro original + refinamientos
├── descargar_datos.py        # Descarga opcional de datos (solo lectura, hashes)
├── requirements.txt
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── tests/
│   └── test_rag_core.py
├── grafo_enriquecido.ttl     # COPIAR del repositorio original
├── embed_matrix_v2.npy      # COPIAR del repositorio original
└── id_map_v2.pkl             # COPIAR del repositorio original
```

## Pruebas

```bash
python -m pytest -q
```

Estas pruebas usan un grafo pequeño sintético, así que no necesitan los archivos
originales ni conexión a Groq para ejecutarse.

## Metodología recomendada para la siguiente etapa

1. Crear un conjunto de 40–80 consultas con resultados lingüísticamente
   validados, incluyendo negaciones, faltantes, familias, polisemia y países.
2. Comparar `BM25`, `MiniLM`, `RRF` por `Recall@5`, `Recall@10` y `MRR@10`.
3. Verificar exactitud de respuestas mediante atribución `claim → triple RDF
   → documento original` y cuantificar afirmaciones no sustentadas.
4. Añadir fecha, fuente, tipo de clasificación y grado de confianza a
   hablantes/vitalidad, en vez de inferir riesgo a partir de un umbral.
5. Auditar la taxonomía del corpus: no todos los registros podrían representar
   de forma inequívoca lenguas indígenas contemporáneas.

**Nota:** El error original no depende de Streamlit: Groq retiró
`llama-3.3-70b-versatile` para planes gratuitos y developer el 16 de agosto
 de 2026. Documentación del proveedor:
https://console.groq.com/docs/deprecations
