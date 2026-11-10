#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lenguas Indígenas · Sudamérica — Streamlit GraphRAG (versión revisada).

Lanzar: streamlit run app_lenguas.py
Datos requeridos en la misma carpeta: grafo_enriquecido.ttl,
  embed_matrix_v2.npy, id_map_v2.pkl.
"""
from __future__ import annotations

import csv
import html
import io
import os
import time
from pathlib import Path

import streamlit as st

from rag_core import (DEFAULT_MODEL, KnowledgeGraph, Retriever, answer_question,
                      detect_country, detect_family, load_vectors,
                      normalize, readable_groq_error)
from styles import CSS

ROOT = Path(__file__).resolve().parent
TTL = ROOT / "grafo_enriquecido.ttl"
MATRIX = ROOT / "embed_matrix_v2.npy"
ID_MAP = ROOT / "id_map_v2.pkl"

st.set_page_config(page_title="Lenguas Indígenas · Sudamérica", page_icon="🗺️",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value or ""), quote=True)


def clean_multiline(text: str) -> str:
    return esc(text).replace("\n", "<br>")


def get_secret(name, fallback="") -> str:
    """Lee primero entorno, luego Secrets de Streamlit (Cloud o instalación local)."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        value = st.secrets.get(name, fallback)
        return str(value).strip() if value is not None else fallback
    except Exception:
        return fallback


def has_groq_key() -> bool:
    key = get_secret("GROQ_API_KEY")
    return bool(key and key.lower() not in {"tu_clave_aqui", "your_api_key", "gsk_xxx"})


@st.cache_resource(show_spinner=False)
def load_graph(path: str, schema_version: str = "relaciones-v5-kariri") -> KnowledgeGraph:
    # versionar caché evita mezclar instancias antiguas con clases nuevas
    return KnowledgeGraph.from_turtle(path)


@st.cache_resource(show_spinner=False)
def load_embeddings(matrix_file: str, idmap_file: str):
    return load_vectors(matrix_file, idmap_file)


@st.cache_resource(show_spinner=False)
def load_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


PREGUNTAS = [
    "¿Qué lenguas indígenas se hablan actualmente en Perú?",
    "¿Qué lenguas se hablan en Bolivia?",
    "¿Qué lenguas aparecen asociadas a Brasil?",
    "¿Qué lenguas pertenecen a la familia quechua?",
    "¿Qué lenguas pertenecen a la familia tupí-guaraní?",
    "¿A qué familia pertenece el jaqaru?",
    "¿En qué país se habla el matsés?",
    "¿Qué lenguas están relacionadas con el quechua?",
    "¿Qué sabemos sobre el asháninka?",
    "¿Qué sabes sobre la lengua Kariri?",
    "¿Cuáles son algunas lenguas con pocos hablantes?",
]


def evidence_view(graph: KnowledgeGraph, ids: list[str]):
    if not ids:
        st.caption("No encontré relaciones suficientes para esta pregunta.")
        return
    for i, eid in enumerate(ids[:12], 1):
        e = graph.entities.get(eid)
        if not e:
            continue
        family_labels = [graph.entities[f].label for f in e.families if f in graph.entities]
        metadata = ["Lengua" if e.kind == "Language" else "Familia o subgrupo"]
        if e.iso:
            metadata.append(f"ISO {e.iso}")
        if e.countries:
            metadata.append(", ".join(e.countries))
        line = f'<div class="evidence"><b>{esc(e.label)}</b> · {esc(" · ".join(metadata))}'
        if family_labels:
            line += f'<div>Familias y subgrupos: {esc(", ".join(family_labels[:5]))}</div>'
        if e.sources:
            links = [f'<a target="_blank" rel="noopener noreferrer" href="{esc(url)}">Fuente {n}</a>'
                     for n, url in enumerate(e.sources[:3], 1)]
            line += '<div>Más información: ' + ' · '.join(links) + '</div>'
        line += '</div>'
        st.markdown(line, unsafe_allow_html=True)


def card(entity: object, graph: KnowledgeGraph):
    from rag_core import Entity
    e: Entity = entity
    metadata = [f"ISO {e.iso}"] if e.iso else []
    metadata.extend(e.countries)
    if e.kind == "Family":
        metadata.append(f"{len(e.members)} lenguas asociadas")
    description = (e.wiki or e.description or e.comment).strip()
    if len(description) > 260:
        description = description[:257].rsplit(" ", 1)[0] + "…"
    families = [graph.entities[f].label for f in e.families if f in graph.entities]
    html_card = f'<div class="lang-card"><div class="name">{esc(e.label)}</div>'
    html_card += f'<div class="meta">{esc(" · ".join(metadata))}</div>'
    if description:
        html_card += f'<div class="desc">{esc(description)}</div>'
    if families:
        html_card += f'<div class="meta" style="margin-top:.55rem">Familia(s): {esc(", ".join(families[:6]))}</div>'
    html_card += '</div>'
    st.markdown(html_card, unsafe_allow_html=True)


st.markdown('''<div class="app-header"><div class="app-header-title">🗺️ Lenguas Indígenas · Sudamérica</div>
<div class="app-header-sub">GraphRAG Explorer</div></div>''', unsafe_allow_html=True)

if not TTL.is_file():
    st.error("Falta el archivo `grafo_enriquecido.ttl` en la carpeta del proyecto.")
    st.stop()

try:
    with st.spinner("Preparando relaciones del grafo…"):
        graph = load_graph(str(TTL), schema_version="relaciones-v5-kariri")
        matrix, ids = (load_embeddings(str(MATRIX), str(ID_MAP))
                       if MATRIX.is_file() and ID_MAP.is_file() else (None, None))
    retriever = Retriever(graph, matrix, ids)
except Exception as error:
    st.error("No se pudieron cargar los archivos del proyecto.")
    st.caption(f"Detalle: {type(error).__name__}. Revisa los archivos de datos.")
    st.stop()

st.markdown('''<p class="intro">Explora el grafo de conocimiento de lenguas indígenas de Sudamérica.
Consulta en lenguaje natural, explora lenguas por familia o país, o revisa la arquitectura del sistema.</p>''',
            unsafe_allow_html=True)

tab_chat, tab_explore, tab_about = st.tabs(["💬 Consultas", "🔎 Explorar", "ℹ️ Acerca de"])

if "messages" not in st.session_state:
    st.session_state.messages = []

with tab_chat:
    mode = st.radio("Modo de pregunta", ["📋 Preguntas sugeridas", "✍️ Pregunta libre"],
                    horizontal=True, label_visibility="collapsed", key="question_mode")
    chosen_model = get_secret("GROQ_MODEL", DEFAULT_MODEL)
    if has_groq_key():
        st.caption(f"🟢 Clave Groq detectada · Modelo: {chosen_model}")
    else:
        st.warning("Falta una API key válida de Groq. Añade GROQ_API_KEY en Streamlit Secrets para activar el LLM.")

    with st.expander("⚙️ Opciones y conexión", expanded=False):
        extended_search = st.checkbox("Búsqueda semántica ampliada (opcional)",
                                      value=False, key="extended_search",
                                      disabled=(matrix is None or ids is None))
        st.caption("La clave no se muestra ni se envía al historial.")
        if st.button("Probar API y modelo", disabled=not has_groq_key(), key="check_groq"):
            try:
                from groq import Groq
                groq_client = Groq(api_key=get_secret("GROQ_API_KEY"),
                                   timeout=6.0, max_retries=0)
                available = {m.id for m in groq_client.models.list().data}
                if chosen_model in available:
                    st.success(f"Groq responde y el modelo {chosen_model} está disponible.")
                else:
                    st.error(f"La clave funciona, pero {chosen_model} no aparece entre los modelos disponibles. "
                             "Cambia GROQ_MODEL por uno accesible en tu cuenta.")
            except Exception as exc:
                st.error(readable_groq_error(exc))

    with st.form("question_form", clear_on_submit=False):
        if mode.startswith("📋"):
            question = st.selectbox("Selecciona una pregunta", [""] + PREGUNTAS,
                                    format_func=lambda s: s or "Selecciona una pregunta…",
                                    label_visibility="collapsed")
        else:
            question = st.text_input("Escribe tu pregunta", label_visibility="collapsed",
                                     placeholder="¿Qué información existe sobre el jaqaru?", max_chars=450)
        submitted = st.form_submit_button("✨ Preguntar", type="primary", use_container_width=True)

    if submitted and question and question.strip():
        question = question.strip()
        st.session_state.messages.append({"role": "user", "text": question})
        with st.spinner("Consultando las lenguas y preparando la respuesta…"):
            started = time.perf_counter()
            # Un Transformer se usa solo en preguntas abiertas: las consultas
            # por país/familia se resuelven por relaciones RDF más rápido.
            retriever.embedder = None
            if extended_search and matrix is not None and ids is not None:
                try:
                    retriever.embedder = load_sentence_model()
                except Exception:
                    pass  # BM25 sigue funcionando; Groq no se desactiva.
            retrieval_ready = time.perf_counter()
            try:
                answer = answer_question(
                    question, graph, retriever,
                    api_key=get_secret("GROQ_API_KEY"),
                    model=chosen_model,
                    top_k=9,
                )
            except Exception as exc:
                from rag_core import Answer
                answer = Answer(
                    "No pude procesar esa consulta. Intenta nuevamente.", [],
                    "Error de consulta", f"Error interno: {type(exc).__name__}."
                )
            finished = time.perf_counter()
        st.session_state.messages.append({
            "role": "assistant", "text": answer.text,
            "evidence": answer.evidence, "method": answer.method,
            "notice": answer.notice,
            "elapsed": finished-started,
            "preparation": retrieval_ready-started,
        })
        st.rerun()
    elif submitted:
        st.warning("Selecciona o escribe una pregunta antes de continuar.")

    if st.session_state.messages:
        st.divider()
        # Orden cronológico: la pregunta aparece antes de su respuesta.
        for message in st.session_state.messages[-20:]:
            if message["role"] == "user":
                st.markdown(f'<div class="msg-user"><div class="lbl">Tú</div>'
                            f'<div class="txt">{clean_multiline(message["text"])}</div></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="msg-bot"><div class="lbl">Asistente</div>'
                            f'<div class="txt">{clean_multiline(message["text"])}</div></div>',
                            unsafe_allow_html=True)
                elapsed = message.get("elapsed")
                timing = f" · {elapsed:.1f} s" if elapsed is not None else ""
                if message.get("notice"):
                    st.warning("Groq no pudo generar una respuesta. Puedes revisar el motivo abajo.")
                with st.expander("🔎 Relaciones consultadas", expanded=False):
                    st.caption(f'{message.get("method", "")} {timing}')
                    if message.get("notice"):
                        st.caption(message["notice"])
                    evidence_view(graph, message.get("evidence", []))
        if st.button("🗑️ Limpiar historial", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("Selecciona o escribe una pregunta para comenzar.")

with tab_explore:
    c1, c2 = st.columns(2)
    with c1:
        kind_label = st.selectbox("Tipo", ["Lenguas", "Familias"])
    with c2:
        countries = sorted({c for e in graph.languages() for c in e.countries})
        chosen_country = st.selectbox("País", ["Todos"] + countries,
                                      disabled=(kind_label == "Familias"))
    fam_options = sorted(graph.families(), key=lambda e: e.label.lower())
    fam_lookup = {f"{f.label} · {f.id}": f.id for f in fam_options}
    selected_family = st.selectbox("Familia lingüística", ["Todas"] + list(fam_lookup),
                                    disabled=(kind_label == "Familias"))
    search = st.text_input("Filtrar por nombre o ISO", placeholder="Quechua, jaqaru, ...")
    candidates = graph.languages() if kind_label == "Lenguas" else graph.families()
    if chosen_country != "Todos" and kind_label == "Lenguas":
        candidates = [e for e in candidates if chosen_country in e.countries]
    if selected_family != "Todas" and kind_label == "Lenguas":
        family_id = fam_lookup[selected_family]
        candidates = [e for e in candidates if family_id in e.families]
    if search:
        from rag_core import normalize
        candidates = [e for e in candidates if normalize(search) in normalize(e.label + " " + e.iso)]
    candidates.sort(key=lambda e: e.label.casefold())
    st.caption(f"{len(candidates)} resultados · mostrando hasta 50")
    for item in candidates[:50]:
        card(item, graph)

    if candidates:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "tipo", "lengua_o_familia", "iso", "paises", "hablantes_sin_fecha", "familias"])
        for item in candidates:
            writer.writerow([item.id, item.kind, item.label, item.iso, "; ".join(item.countries),
                             "" if item.speakers is None else item.speakers,
                             "; ".join(item.families)])
        st.download_button("⬇️ Descargar selección como CSV", output.getvalue().encode("utf-8-sig"),
                           file_name="seleccion_lenguas_rag_sa.csv", mime="text/csv",
                           use_container_width=True)

with tab_about:
    stats = graph.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Lenguas y variedades", stats["languages"])
    c2.metric("Familias y subgrupos", stats["families"])
    c3.metric("Lenguas con descripción", stats["with_comment"])

    st.markdown('''<div class="about-sec"><h4>🗺️ Explora las relaciones</h4>
<p>Este explorador conecta <strong>lenguas, países y familias lingüísticas</strong>.
Puedes preguntar qué lenguas aparecen asociadas a un país, a qué familia
pertenece una lengua o qué variedades comparten una agrupación.</p>
<p>Las lenguas de una misma familia están relacionadas lingüísticamente.
La semejanza entre sus descripciones no demuestra parentesco.</p></div>''', unsafe_allow_html=True)
    st.markdown('''<div class="about-sec"><h4>💬 ¿Cómo responde?</h4>
<p>Primero se encuentran relaciones en el grafo y fichas relevantes con búsqueda léxica.
Puedes activar una búsqueda semántica ampliada en las opciones.
Después, Groq explica los resultados en lenguaje natural y puede decir
que no sabe cuando faltan datos. Las cifras de hablantes, si se solicitan,
se tratan como estimaciones orientativas.</p></div>''', unsafe_allow_html=True)
    st.markdown('''<div class="chips"><span class="chip green">Lengua → Familia</span>
<span class="chip green">Lengua → País</span><span class="chip">RDFLib</span>
<span class="chip">BM25</span><span class="chip">MiniLM opcional</span>
<span class="chip">Groq</span></div>''', unsafe_allow_html=True)

st.markdown('''<div class="tfooter"><p>🗺️ <b>Lenguas Indígenas · Sudamérica</b> · GraphRAG Explorer</p>
<p>Corpus RDF · Recuperación verificable · Fuentes por entidad</p></div>''', unsafe_allow_html=True)
