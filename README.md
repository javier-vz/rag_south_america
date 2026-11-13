# RAG South America v7.1.1 — Corrección de traducciones

**Corrección exclusiva de compatibilidad de textos.** La v7.1.1 no altera el grafo, Groq, la navegación, las relaciones visuales ni la apariencia.

## Qué causaba el error

`app_lenguas.py` v7.1 llama a `explore_search_placeholder`. Si Streamlit ejecuta un `locales.py` anterior (v7.0 o v6.9), la clave no existe y se produce un `KeyError` antes de abrir Explorar. En `locales.py` v7.1 **la clave ya estaba bien definida**.

## Solución

`app_lenguas.py` incorpora traducciones de respaldo para las 20 nuevas claves de la interfaz agregadas después de v6.9, en español, portugués e inglés. Se utiliza `locales.py` como primera fuente; el respaldo solo se usa si falta una clave. Esto evita que una actualización parcial de archivos derribe la aplicación.

## Actualización

Para aplicar la corrección desde v7.1, sustituye `app_lenguas.py` y `locales.py` de forma conjunta. El ZIP mínimo incluye ambos archivos y una prueba de regresión; los demás módulos no cambian. En Streamlit Cloud, reinicia la aplicación si persiste un módulo anterior en memoria.

No subas `secrets.toml` ni tu clave Groq a GitHub.

## Pruebas

`python -m pytest -q tests`

La prueba nueva reproduce el caso con `locales.py` v7.0 y v6.9 y comprueba los tres idiomas. Se mantienen las demás pruebas del proyecto.
