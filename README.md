# RAG South America v7.1 — Explorar más sencillo

Versión basada en v7.0, que incluye la búsqueda por Glottocode y simplifica los recorridos de «Explorar».

## Qué cambia

- **Búsqueda primero:** se escribe nombre, Glottocode o ISO sin seleccionar un país. La búsqueda se hace en todo el grafo, aunque se haya usado antes un filtro de país.
- **Una coincidencia:** se abre directamente la ficha y la red local, sin un desplegable adicional.
- **Varias coincidencias:** se elige una de una lista corta con nombre y Glottocode.
- **Navegación opcional:** país, agrupación y, solo si se solicita, subgrupo. Puedes escoger una lengua del país sin seleccionar antes su familia.
- **Kariri:** `kari1254` se encuentra por nombre o Glottocode aunque su ficha no tenga una relación RDF de país; no la añadimos artificialmente a Brasil.
- **Identificador principal:** Glottocode en fichas y exportaciones; ISO se mantiene cuando existe.
- Tres idiomas: español, portugués e inglés.

## Qué NO cambia

Ni Groq ni el motor de respuestas (salvo reconocer Glottocodes en preguntas), el grafo RDF, los datos, la red local de relaciones, las fuentes ni el diseño CSS. GitHub no fue modificado.

## Actualizar desde v7.0

Reemplaza `app_lenguas.py`, `explore_tree.py`, `rag_core.py`, `locales.py`. Los otros archivos permanecen iguales. La carpeta `.streamlit` conserva su configuración y la API key se mantiene en Streamlit Secrets.

Ejecutar: `streamlit run app_lenguas.py`.
