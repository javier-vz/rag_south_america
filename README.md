# RAG_SA v6.5 — Exploración por país, agrupación y lengua

Esta actualización mantiene **el estilo oscuro/verde** y conserva el motor de consultas, el modelo configurado en Groq y los archivos RDF. Introduce únicamente una navegación más compacta en «Explorar» y los textos solicitados en español y portugués.

## Cambios

- **Exploración encadenada:** elige un país, una agrupación principal, un subgrupo opcional y una lengua. Las opciones se limitan a los registros relacionados con los filtros anteriores; ya no se abre una lista plana con todas las familias y subfamilias.
- **Búsqueda directa:** también puedes buscar una lengua por nombre o código ISO, incluso sin seleccionar un país.
- **Lenguas sin familia:** quedan accesibles mediante «Sin agrupación registrada», en vez de desaparecer.
- **Presentación:** explica que el proyecto surge para conversar con datos lingüísticos especializados, especialmente a partir de Glottolog. Describe las aportaciones de Wikipedia y Wikidata y señala que Ethnologue aparece citado en algunas descripciones, **sin atribuirle una importación separada**.
- **Enlaces:** Glottolog, Wikipedia, Wikidata y Ethnologue tienen sus enlaces visibles en «Acerca de / Sobre o projeto».
- **Responsable y sugerencias:** Javier Maximiliano Vera Zúñiga — jxvera@gmail.com.

### Nota sobre «agrupación» y «subgrupo»

El grafo original registra relaciones entre lenguas y agrupaciones, pero no garantiza una jerarquía explícita entre todos los nodos de familia. Para navegar de lo general a lo específico, los selectores comparan **conjuntos de lenguas asociadas**. Es solo una organización de menús, no una afirmación taxonómica nueva ni una modificación de las relaciones RDF.

## Actualizar desde v6.4

Reemplaza `app_lenguas.py` y `locales.py` y **añade** `explore_tree.py` en la misma carpeta del proyecto. No necesitas cambiar `rag_core.py`, `styles.py`, `requirements.txt`, los tres archivos de datos ni tus secretos de Streamlit.

## Instalar desde el ZIP completo

```bash
python descargar_datos.py
pip install -r requirements.txt
streamlit run app_lenguas.py
```

Configura `GROQ_API_KEY` en los secretos de Streamlit. No la incluyas en GitHub.

## Pruebas

```bash
python -m pytest -q
```

Las pruebas abarcan recuperación, los idiomas, el envío de preguntas y los filtros anidados. La aplicación desplegada con datos y clave de producción todavía requiere comprobación manual.
