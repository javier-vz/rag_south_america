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

   Modelo predeterminado (menor latencia): `openai/gpt-oss-20b`. Si prefieres priorizar capacidad sobre rapidez, cambia GROQ_MODEL a `openai/gpt-oss-120b`.
   Puedes cambiarlo con `GROQ_MODEL` si el proveedor actualiza su catálogo.

   **Con Groq:** todas las preguntas pasan por el LLM, incluyendo recuentos y listados.
   Los cálculos RDF proporcionan hechos verificables al LLM, pero NO sustituyen su respuesta.

   **Sin Groq:** se muestra una orientación breve y una advertencia visible con la causa. Esto NO se presenta como generación del LLM.
   Siguen funcionando los recuentos, filtros, exploración y búsqueda. Sin `sentence-transformers` disponible
   tras instalar dependencias o sin conexión para descargar el modelo por primera
   vez, la recuperación usa BM25 de respaldo.

## Arquitectura de respuesta

```
Pregunta libre o sugerida
   ↓
Recuperación rápida BM25 + relaciones RDF (por defecto)
   o BM25 + embeddings MiniLM + RRF (opción ampliada)
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
| Recuperación | Suma de puntuaciones de escalas distintas | BM25 rápido; RRF con MiniLM opcional |
| Consulta de país/familia | Ajustes ad hoc al contexto | Filtros y recuentos RDF + redacción Groq en todas las respuestas |
| Cifras de hablantes | Usadas para inferir vitalidad | Faltantes y ceros explícitos; sin afirmar extinción |
| Trazabilidad | Texto sin respaldo visual | IDs `[G1]` y enlaces a fuentes del grafo |
| Fallo del LLM | Error expuesto como respuesta | Respaldo etiquetado (no se hace pasar por respuesta de Groq) |
| Navegación | Fichas de 50 registros | Filtros + exportación CSV |
| HTML de la UI | Interpolación no escapada | `html.escape` |
| Código | Un archivo monolítico | Entrada + motor + estilos + tests |

## Limitaciones que siguen abiertas

- El grafo contiene fichas no necesariamente indígenas (p. ej., español amazónico peruano y criollos) y agrupaciones familiares jerárquicas: no se deben tratar los 552 o 409 nodos como inventarios oficiales.
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
├── auditar_grafo.py          # Reporte reproducible JSON + CSV de calidad
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

## Estilo de respuesta, velocidad y honestidad

- Respuesta por defecto: **un párrafo claro, cálido y breve**, sin explicaciones técnicas innecesarias.
- Solo presenta cifras de hablantes si las preguntas explícitamente. Son estimaciones sin fecha común, **no** un censo.
- Si falta evidencia, el LLM puede decir **«No lo sé con seguridad»** y explicar por qué sin inventar datos.
- Groq redacta todas las respuestas cuando está disponible. La evidencia se muestra aparte en **Fuentes y detalles**.
- Modelo por defecto `openai/gpt-oss-20b` con razonamiento bajo, un solo intento y timeout de 9 segundos. Puede seguir demorando según red/cuota; no existe tiempo fijo garantizado.
- La búsqueda semántica ampliada es **opcional**, para no descargar o cargar un Transformer grande en la primera pregunta. En modo rápido se usa RDF + BM25 + LLM.
- La confirmación de que la clave funciona se realiza en **Opciones y conexión → Probar API y modelo**. No publiques tu API key ni la compartas aquí.
- No se modificó el repositorio de GitHub.

## Calidad del corpus y respuesta esperada

Para leer el diagnóstico detallado, abre `AUDITORIA_GRAFO_2026-11-09.md`. Allí están los problemas reales detectados en el TTL, incluidos los **325 registros sin dato de hablantes**, **91 sin país**, **63 sin familia** y **2457 referencias externas embebidas en URIs locales**.

Para reproducir la auditoría sin modificar el grafo:

```bash
python auditar_grafo.py
```

Este comando crea `auditoria_datos.json` y `auditoria_lenguas.csv` con alertas para revisión manual. Nada se modifica en GitHub.

En la pestaña Consultas, para preguntas sencillas como «¿Qué lenguas se hablan en Perú?», el comportamiento esperado es un **párrafo amistoso**, con ejemplos y sin estadísticas innecesarias. Las preguntas sobre cifras reciben solo estimaciones y su contexto de incertidumbre. Cuando los datos son insuficientes, el asistente puede decir «no lo sé con seguridad».

### Configuración de velocidad

El modo rápido predeterminado usa BM25 y el grafo y evita cargar un modelo Transformer durante una pregunta por país o familia. En **Opciones y conexión** puedes activar «Búsqueda semántica ampliada» (BM25 + MiniLM + RRF) para consultas abiertas donde la similitud semántica añade valor. La primera activación de MiniLM puede tardar más. Groq sigue generando todas las respuestas cuando está disponible.

Modelo Groq predeterminado `openai/gpt-oss-20b`, tiempo de espera máximo por llamada 9 s, sin reintentos automáticos. Esto **no** garantiza una latencia total menor a 9 s en todos los despliegues.

### Alcance de validación

Las pruebas incluidas usan datos sintéticos y respuestas de Groq simuladas; no se han ejecutado llamadas reales con la clave de la persona ni una evaluación humana de respuestas. El corpus original no se redistribuye en el ZIP: se descarga con `python descargar_datos.py` desde el repositorio público y se verifica su hash.
