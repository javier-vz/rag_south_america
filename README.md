# RAG South America — v7.0

**Nuevo: relaciones visuales en una red local interactiva**

Esta versión continúa las mejoras de **v6.8 (preguntas de seguimiento)** y **v6.9 (navegación contextual desde las respuestas)**. Conserva el estilo oscuro y verde, la interfaz en español, portugués e inglés, el explorador anidado y **Groq como generador de respuestas**.

## Qué añade la v7.0

Dentro de **Explorar** aparece una red local cuando se abre una ficha de lengua o familia:

- **La ficha consultada aparece en verde**, en el centro del gráfico.
- **Las agrupaciones lingüísticas aparecen en amarillo**, conectadas con la lengua solo cuando la asociación `belongsToFamily` está en el RDF.
- **Los países aparecen en gris**, cuando existe la asociación de país en el registro.
- **Otras lenguas aparecen en azul**, conectadas a una agrupación común, no directamente entre ellas: esto representa pertenencia compartida, **no** una nueva afirmación de parentesco.

Puedes **seleccionar un nodo** en el gráfico y pulsar **«Abrir ficha en Explorar»**; si se trata de otra lengua o agrupación, también puedes **centrar la red** allí. Existe un selector equivalente bajo el gráfico para pantallas pequeñas o cuando sea difícil pulsar un nodo. El gráfico permite arrastrar y ampliar la vista mediante los controles habituales de Plotly.

La red aparece tanto al llegar desde los botones de la conversación como al seleccionar una lengua en el explorador. No añade una cuarta pestaña ni sustituye la navegación que ya funciona.

### Un grafo deliberadamente pequeño

Por defecto muestra **hasta tres agrupaciones, dos países y seis lenguas adicionales** en una ficha de lengua; en una ficha de familia, **hasta nueve lenguas asociadas**. Si hay más registros, lo indica debajo del gráfico.

La red se construye a partir de las relaciones disponibles. **No utiliza `semanticallySimilarTo` como parentesco**, no interpreta el texto de Wikipedia para inventar países o familias y no crea nuevas relaciones jerárquicas. Por ejemplo, Kariri tiene una descripción histórica en el corpus, pero puede no tener relaciones estructuradas de país o familia: en ese caso la red muestra únicamente su ficha y explica la ausencia de enlaces.

## Actualización mínima desde v6.9

Sustituye juntos:

1. `app_lenguas.py`
2. `locales.py`
3. `requirements.txt`

Y agrega el nuevo módulo **`relation_graph.py`**. Se incluye también una prueba opcional: `tests/test_relation_graph_v70.py`.

**No hace falta modificar** `rag_core.py`, `conversation_nav.py`, `explore_tree.py`, `styles.py`, los archivos RDF, los embeddings ni tu clave Groq.

El proyecto ahora requiere `plotly>=6,<7` para dibujar la red. Tras copiar los archivos:

```bash
pip install -r requirements.txt
streamlit run app_lenguas.py
```

Si estás comenzando desde el ZIP completo de código, descarga primero los archivos de datos con `python descargar_datos.py` o reutiliza los que ya tienes en el repositorio. **Los ZIP no vuelven a distribuir el grafo ni los embeddings**; permanecen sin cambios.

## Verificación

```bash
python -m pytest -q
```

**116 pruebas aprobadas** en el entorno de preparación. Se incluyen regresiones de las 95 pruebas anteriores y casos nuevos con enlaces válidos, familias grandes, fichas sin relaciones, límite de nodos, selección interactiva y traducciones.

### Recorrido manual recomendado

1. Pregunta **«¿Qué sabes sobre Jaqaru?»**. Debajo de la respuesta usa **«Ver ficha»** y, en **Explorar**, abre «Relaciones de Jaqaru» (si no está ya desplegado).
2. Selecciona el nodo de la familia **Aymaran** o el país **Perú**. Abre la ficha, o céntrala si es una agrupación.
3. Desde una lengua adicional de la red, pulsa **«Centrar red aquí»**; comprueba que cambia el foco de la visualización.
4. Prueba **Kariri**: la descripción de la ficha sigue disponible, pero la red no debe inventar un vínculo con Brasil o con una familia a partir del texto.
5. Comprueba las mismas acciones con **Português** y **English**, y en una pantalla estrecha usando el selector de nodos.

**Limitación:** se validó la construcción del grafo y la figura de Plotly mediante pruebas automatizadas; todavía es necesario comprobar los clics y el diseño responsive en el despliegue real de Streamlit. No se realizó una consulta real con tu API key.

## Archivos añadidos o modificados

- **Modificados:** `app_lenguas.py`, `locales.py`, `requirements.txt`, `README.md`.
- **Nuevos:** `relation_graph.py`, `tests/test_relation_graph_v70.py`.
- **Sin cambios:** las demás fuentes del proyecto y sus datos.

**No se modificó GitHub.**
