# Auditoría técnica y lingüística de RAG_SA

Fecha de revisión: 9 de noviembre de 2026.
Rama pública consultada: `main`, commit `6f4ce3c4ea78c002bceab0adbd370275152886f5`.
Código original: `https://github.com/javier-vz/RAG_SA`.

## Hechos comprobados en el repositorio original

| Aspecto | Observación |
|---|---|
| Archivo principal | `app_lenguas.py` (~36,7 KB); lógica, interfaz y CSS mezclados |
| RDF | `grafo_enriquecido.ttl` (~1,28 MB) |
| Índice existente | `embed_matrix_v2.npy`, `id_map_v2.pkl`, `index_faiss_v2.faiss` |
| Entidades | 552 instancias `ex:Language`, 409 instancias `ex:Family` |
| Comentarios | 491 de 552 lenguas tienen `rdfs:comment` |
| Cifras de hablantes | 212 con valor positivo; 15 con valor cero; 325 sin valor |
| Registros asociados a Perú | 86; 42 con valor positivo, 3 con cero, 41 sin cifra |
| Modelo Groq del original | `llama-3.3-70b-versatile` — retirado para tiers free/developer |
| Lectura TTL | Separación por regex de bloques `ex:` en vez de parser RDF |
| Recuperación | Producto matricial con embeddings; tokens coincidentes; suma 60/40 |
| Uso de FAISS | Hay índice en el repositorio, pero `app_lenguas.py` consulta la matriz directa, no el índice FAISS |
| Recuento regional | Rama especial devuelve el número de nodos del grafo como si fueran lenguas reales en Sudamérica |
| Trazabilidad | No presenta al usuario IDs de evidencia ni relación afirmación→fuente |
| Pruebas | No había directorio de pruebas; `README.md` solo contenía `# RAG_SA` |

**Nota:** Los conteos describen el archivo y no equivalen necesariamente a inventarios
lingüísticos actuales. Las etiquetas `Family` pueden incluir nodos de distintos niveles
clasificatorios, no 409 familias de nivel equivalente.

## Prioridad 0 — Error visible

**P0. Cambiar el modelo de Groq.** El modelo usado dejó de estar disponible para
el nivel gratuito/desarrollador el 16/08/2026. En la actualización, el modelo
se elige desde `GROQ_MODEL` (por defecto `openai/gpt-oss-20b`).

**P0. Respuestas naturales con LLM para todas las preguntas.** Antes, el 404 se mostraba
como mensaje del asistente. Ahora el RDF calcula hechos y Groq redacta el texto
final para consultas estructuradas y abiertas. Si una llamada falla la UI ofrece
un aviso explícito y una orientación breve, sin disfrazarla de respuesta del LLM.

## Prioridad 1 — Validez epistemológica

**P1. No inferir riesgo de extinción a partir de umbrales de hablantes.**
En el original, 0 significa "extinta", menos de 100 "crítica",
100–999 "amenazada" y 1.000+ "activa". Estas etiquetas no tienen en el código
una referencia a UNESCO, EGIDS u otra escala con metodología y año.
La versión nueva elimina esa conversión y distingue faltante/0/positivo.

**P1. Corregir consultas sobre lenguas "actuales".**
En el original, al filtrar Perú se exige un número de hablantes y se excluyen
registros sin cifra o que un texto en inglés parezca describir como históricos.
Esto sesga los resultados. La nueva versión separa explícitamente 86 registros,
42 con cifra positiva, 41 sin cifra y 3 con cero. No certifica actualidad.

**P1. No presentar totales del corpus como censos.**
`ex:Language` incluye datos de diversas fuentes, distintos niveles de
clasificación y al menos un registro llamado "Peruvian Amazonian Spanish".
Se necesita auditar qué significa "lengua indígena" en la ontología.

**P1. Auditoría de fuentes y datación.** La ontología incluye
`ex:wikidata_speakers` como texto numérico sin fecha homogénea, enlaces
a fuentes en `ex:describedByResource` y comentarios enriquecidos con IA.
La nueva interfaz distingue esos planos, pero una auditoría científica final
debe revisar en la fuente primaria cada número, fecha y clasificación.

## Prioridad 2 — Recuperación y grafo

**P2. Parser RDF real.** Los bloques Turtle pueden tener predicados
multilínea, IRIs y literales complejos; regex es frágil. `rdflib` permite
interpretar correctamente triples, relaciones y URI.

**P2. BM25 + RRF.** La suma de una similitud semántica de rango variable
con un puntaje léxico normalizado por máximo puede cambiar arbitrariamente
según la pregunta. Reciprocal Rank Fusion fusiona rankings, no escalas.
Se conserva el MiniLM multilingüe y los vectores `.npy` originales.

**P2. Estructuración de consultas.** Consultas "cuántas", "de Perú"
o "de la familia Quechuan" se resuelven consultando entidades RDF.
Preguntas explicativas abiertas pasan por recuperación + generación.

**P2. Relaciones explícitas.** `belongsToFamily` sirve para construir miembros
por familia. `semanticallySimilarTo` refiere similitud de embeddings del
corpus: NO es prueba de parentesco, contacto ni similitud tipológica.

**P2. Estructura multinivel de familias.** Un idioma puede relacionarse
con familias de varios niveles. La siguiente etapa debe revisar jerarquías
y distinguir familia, rama, subgrupo, idioma y variedad.

## Prioridad 3 — Producto y operación

- Conservar el estilo GitHub-dark, tipografía Inter, tarjetas y verde.
- Escapar textos de usuario y RDF cuando se interpolan en HTML.
- Mostrar IDs, ISO y enlaces externos junto a cada respuesta.
- Añadir filtros por familia + exportación CSV sin modificar los datos.
- Centralizar alias y eliminar duplicaciones en el módulo de recuperación.
- Crear pruebas automatizadas de parser, razonamiento, fallbacks y ranking.
- Simplificar dependencias: FAISS no es obligatorio para ~961 entidades.
- Separar código del estilo para permitir cambios sin romper el RAG.

## Qué se ha implementado en este ZIP

1. `rag_core.py`: RDFLib, entidades, BM25, fusión RRF, generador Groq,
   funciones para respuestas estructuradas y manejo de fuentes.
2. `app_lenguas.py`: conserva 3 pestañas y la apariencia, añade evidencia
   y modos de respaldo.
3. `styles.py`: paleta y ajustes visuales.
4. `descargar_datos.py`: descarga solo-lectura de los tres assets públicos,
   comprueba SHA1 Git contra commit auditado, nunca modifica GitHub.
5. `tests/test_rag_core.py`: pruebas sobre un grafo sintético reproducible.

**No está verificado todavía en producción**: una ejecución real en
Streamlit Cloud con los tres archivos originales, la clave Groq de la cuenta
y la descarga/carga de MiniLM. Las pruebas automatizadas se ejecutaron sin
API externa; no prueban exactitud científica del contenido completo.

## Próximo experimento científico recomendado

Preparar 60 preguntas anotadas por especialistas:

- 15 consultas sobre lengua concreta;
- 10 consultas de países, verificando faltantes;
- 10 consultas de familia/subfamilia;
- 10 de vitalidad con preguntas imposibles;
- 10 de comparación multi-entidad;
- 5 de desambiguación, contraejemplos y negaciones.

Comparar BM25, MiniLM y RRF con `Recall@5`, `MRR@10`, calidad de citas,
porcentaje de afirmaciones sustentadas y abstención correcta ante falta
 de evidencia. Fijar versiones de datos y registrar latencia/consumo Groq.
Esto separa una demo convincente de un sistema GraphRAG auditable
como producto de investigación.
