# RAG_SA — Auditoría del grafo y decisiones de diseño

**Fuente:** `grafo_enriquecido.ttl` de la rama pública `main` del repositorio `javier-vz/RAG_SA`, consultada el **9 de noviembre de 2026**. Este informe describe el corpus, **no** el inventario real de todas las lenguas indígenas de Sudamérica.

## 1. Inventario calculado sobre el TTL

| Campo comprobado | Cantidad |
|---|---:|
| Nodos `ex:Language` | 552 |
| Nodos `ex:Family` | 409 |
| Entidades entre ambas clases | 961 |
| Lenguas con `rdfs:comment` | 491 |
| Lenguas con resumen Wikipedia | 480 |
| Lenguas con ISO registrado | 466 |
| Lenguas sin valor `wikidata_speakers` | 325 |
| Lenguas con valor de hablantes igual a `0` | 15 |
| Lenguas con valor positivo de hablantes | 212 |
| Lenguas sin país (`wikidata_countries`) | 91 |
| Lenguas sin vínculo `belongsToFamily` | 63 |
| Fichas asociadas a Perú | 86 |
| Fichas de Perú sin cifra de hablantes | 41 |
| Fichas de Perú con cero registrado | 3 |
| Objetos de `describedByResource` con URL externa embebida bajo namespace local | 2457 |

> Algunos recuentos se basan en presencia de propiedad RDF, no en verificación externa. La utilidad `auditar_grafo.py` produce un JSON y un CSV reproducibles sobre el TTL que se copie en la carpeta.

## 2. Riesgos constatados y correcciones propuestas

### A. No todo `ex:Language` es una lengua indígena contemporánea

Ejemplo real del corpus: **Peruvian Amazonian Spanish**, catalogado como `ex:Language` con país Perú. También hay fichas de criollos y del **Classical Quechua** sin asociación territorial. Por lo tanto:

- `552` **no** debe presentarse como total real de lenguas indígenas actuales.
- `86` **no** es una cifra oficial de lenguas indígenas en uso en Perú.
- La nueva app marca solamente etiquetas evidentes para **revisión manual**, sin borrar ni alterar ninguna ficha. Estas marcas son un filtro conservador de presentación, no una clasificación científica.

### B. Las cifras de hablantes son orientativas

325 de 552 fichas no contienen número de hablantes y las cifras presentes no incluyen fecha y método comparables en el modelo actual. Por eso:

- Nunca deducir extinción de `0` o de la ausencia de cifra.
- No convertir un umbral de hablantes en una clasificación de vitalidad.
- Si la pregunta **no** pide cifras, no enviarlas al LLM.
- Si pide cifras, se pueden ofrecer como **estimaciones aproximadas** y no como datos exactos actuales.

### C. Familia es una jerarquía, no una lista plana de familias independientes

El grafo usa hasta varios nodos en `belongsToFamily` para una sola lengua. Por ejemplo, Akuntsu aparece vinculada a múltiples nodos de distintos niveles; existen agrupaciones Tupian, Tupi-Guarani, etc. Por eso **409 no significa 409 familias no solapadas**. La app dice «familias y subgrupos» al describir estos nodos y conserva las asociaciones sin colapsarlas incorrectamente.

### D. País y coordenada son datos parciales

La mayoría de las fichas usa un solo país. Ello **no** prueba que una lengua no cruce fronteras; una coordenada tampoco describe el área completa de distribución. Evitar responder a consultas de territorios específicos (por ejemplo, «Amazonía peruana») como si todo lo clasificado como Perú perteneciera automáticamente a esa subregión.

### E. Diferenciar evidencia de texto generado

`rdfs:comment` fue enriquecido mediante IA en el pipeline original. Puede servir para recuperación, pero **no equivale a corroboración independiente**. Se prefiere el resumen documental disponible; se distingue explícitamente una descripción generada cuando es la única información textual disponible. El usuario puede abrir los enlaces registrados por ficha.

### F. Las referencias externas tienen un error estructural de URI

Ejemplo observado:

```ttl
ex:describedByResource <http://example.org/lang/https://en.wikipedia.org/wiki/Abip%C3%B3n_language>
```

El navegador no puede usar esa dirección como enlace a Wikipedia. La nueva app **normaliza el enlace en memoria**, con `safe_url`, sin modificar el TTL. En una futura limpieza del corpus se recomienda arreglar las URIs de origen en el pipeline y conservar una relación documentada entre fuente y afirmación.

### G. La similitud semántica no es parentesco lingüístico

`semanticallySimilarTo` parece relacionar entidades a partir de cercanía de embeddings y **no** demuestra parentesco, inteligibilidad mutua, transmisión histórica ni familia común. Además, varios `similarityScore` coexisten sin una estructura RDF que asigne inequívocamente cada puntuación a su destino: hay que reificar la relación o representarla como nodo antes de interpretarla como evidencia cuantitativa por enlace.

## 3. Cómo debe contestar el asistente

**Pregunta:** «¿Qué lenguas indígenas se hablan actualmente en Perú?»

**Ejemplo de respuesta deseada:** «En el Perú se hablan numerosas lenguas indígenas. Entre las más conocidas están distintas variedades del quechua y del aimara, y también lenguas amazónicas como el asháninka, el aguaruna, el shawi, el matsés o el yine. Cada una tiene su propia historia y comunidades de hablantes. La lista es solo una muestra de la diversidad lingüística del país».

Los nombres del ejemplo corresponden a fichas observadas del grafo. **Es un ejemplo editorial, no una respuesta generada por Groq** ni una certificación contemporánea independiente. La app pide al LLM una redacción variable, no un texto fijo.

**Si la evidencia es insuficiente:** «No lo sé con seguridad. Aquí hay información sobre las lenguas, pero no datos actualizados y contrastables para responder eso con precisión».

Evitar hablar de «registros», «ABox», porcentajes de cobertura, número exacto de hablantes o «respaldo RDF» en el cuerpo de una respuesta cotidiana.

## 4. Rendimiento y fallos de Groq

La versión anterior cargaba un modelo de embeddings de Hugging Face incluso cuando la pregunta se resolvía directamente con países o familias. Además, los reintentos y tiempos de espera de la API podían alargar la consulta. En la nueva versión:

- **Modo rápido predeterminado:** BM25 y relaciones RDF, sin cargar MiniLM.
- **Modo semántico opcional:** MiniLM + BM25, con costo de carga inicial.
- **Groq** redacta **todas** las respuestas cuando está configurado; motor por defecto `openai/gpt-oss-20b` (cambiable).
- Límite de espera de la petición a Groq de **9 segundos**, sin reintentos ocultos. No es una garantía de respuesta global menor a 9 segundos (la carga/ejecución local también consume tiempo).
- Fallo de API: mostrar el diagnóstico accesible y, si hay datos para ello, una orientación breve **señalada como producida sin Groq**. No disfrazar el fallo como si el LLM hubiese respondido.

## 5. Lo que falta para considerar validado el contenido

Próxima investigación de datos: añadir **fecha de observación**, **fuente de cada afirmación**, **tipo de nodo lingüístico**, **cobertura geográfica con múltiples países**, **vitalidad de fuente especializada**, y evaluar si `rdfs:comment` contiene afirmaciones no documentadas. Después, construir una batería de preguntas humanas con respuestas de referencia y verificar tanto recuperación como fidelidad del LLM. Las pruebas incluidas aquí verifican comportamiento de software; no sustituyen esa evaluación.
