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
from locales import LANGUAGES, QUESTIONS, country_label, t

ROOT = Path(__file__).resolve().parent
TTL = ROOT / "grafo_enriquecido.ttl"
MATRIX = ROOT / "embed_matrix_v2.npy"
ID_MAP = ROOT / "id_map_v2.pkl"

st.set_page_config(page_title="Lenguas indígenas / Línguas indígenas", page_icon="🗺️",
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


def evidence_view(graph: KnowledgeGraph, ids: list[str], lang: str):
    if not ids:
        st.caption(t(lang, "evidence_empty"))
        return
    for eid in ids[:12]:
        e = graph.entities.get(eid)
        if not e:
            continue
        family_labels = [graph.entities[f].label for f in e.families if f in graph.entities]
        metadata = [t(lang, "type_language") if e.kind == "Language" else t(lang, "type_family")]
        if e.iso:
            metadata.append(f"ISO {e.iso}")
        if e.countries:
            metadata.append(", ".join(country_label(lang, c) for c in e.countries))
        line = f'<div class="evidence"><b>{esc(e.label)}</b> · {esc(" · ".join(metadata))}'
        if family_labels:
            line += f'<div>{esc(t(lang,"families_related"))}: {esc(", ".join(family_labels[:5]))}</div>'
        if e.sources:
            links = [f'<a target="_blank" rel="noopener noreferrer" href="{esc(url)}">{esc(t(lang, "source", n=n))}</a>'
                     for n, url in enumerate(e.sources[:3], 1)]
            line += '<div>' + esc(t(lang, "more_info")) + ': ' + ' · '.join(links) + '</div>'
        line += '</div>'
        st.markdown(line, unsafe_allow_html=True)


def card(entity: object, graph: KnowledgeGraph, lang: str):
    from rag_core import Entity
    e: Entity = entity
    metadata = [f"ISO {e.iso}"] if e.iso else []
    metadata.extend(country_label(lang, country) for country in e.countries)
    if e.kind == "Family":
        metadata.append(t(lang, "family_meta", n=len(e.members)))
    families = [graph.entities[f].label for f in e.families if f in graph.entities]
    # Las descripciones originales del RDF pueden estar en inglés.
    # Para que TODO el explorador sea bilingüe, la tarjeta resume solo
    # relaciones directas disponibles, sin traducir ni inventar los textos fuente.
    if e.kind == "Family":
        description = t(lang, "card_family", n=len(e.members))
    else:
        location = (t(lang, "card_location", countries=", ".join(country_label(lang, c) for c in e.countries))
                    if e.countries else t(lang, "card_no_location"))
        fam = (t(lang, "card_family_suffix", families=", ".join(families[:3])) if families else "")
        description = t(lang, "card_language", location=location, family=fam)
    html_card = f'<div class="lang-card"><div class="name">{esc(e.label)}</div>'
    html_card += f'<div class="meta">{esc(" · ".join(metadata))}</div>'
    html_card += f'<div class="desc">{esc(description)}</div>'
    if families:
        html_card += f'<div class="meta" style="margin-top:.55rem">{esc(t(lang, "family_label"))}: {esc(", ".join(families[:6]))}</div>'
    html_card += '</div>'
    st.markdown(html_card, unsafe_allow_html=True)


# Selector bilingüe. Nombres de lenguas y datos RDF se mantienen sin cambios.
lang = st.session_state.get("lang", "es")
st.markdown(f'''<div class="app-header"><div class="app-header-title">🗺️ {esc(t(lang, 'header'))}</div>
<div class="app-header-sub">{esc(t(lang, 'subtitle'))}</div></div>''', unsafe_allow_html=True)

intro_col, language_col = st.columns([4, 1.3], vertical_alignment="center")
with intro_col:
    st.markdown(f'<p class="intro">{esc(t(lang, "intro"))}</p>', unsafe_allow_html=True)
with language_col:
    selected_lang = st.selectbox("🌐 Idioma / Idioma", list(LANGUAGES),
                                 format_func=lambda value: LANGUAGES[value],
                                 index=list(LANGUAGES).index(lang), key="lang_choice")
if selected_lang != lang:
    st.session_state.lang = selected_lang
    st.rerun()

if not TTL.is_file():
    st.error(t(lang, "missing_ttl"))
    st.stop()

try:
    with st.spinner(t(lang, "preparing")):
        graph = load_graph(str(TTL), schema_version="relaciones-v5-kariri")
        matrix, ids = (load_embeddings(str(MATRIX), str(ID_MAP))
                       if MATRIX.is_file() and ID_MAP.is_file() else (None, None))
    retriever = Retriever(graph, matrix, ids)
except Exception as error:
    st.error(t(lang, "load_error"))
    st.caption(t(lang, "load_error_detail", name=type(error).__name__))
    st.stop()

tab_chat, tab_explore, tab_about = st.tabs(
    [t(lang, "tab_chat"), t(lang, "tab_explore"), t(lang, "tab_about")])

if "messages" not in st.session_state:
    st.session_state.messages = []

with tab_chat:
    mode = st.radio(t(lang, "mode"), ["suggested", "free"],
                    format_func=lambda value: t(lang, "suggested_mode" if value == "suggested" else "free_mode"),
                    horizontal=True, label_visibility="collapsed", key="question_mode_v6")
    chosen_model = get_secret("GROQ_MODEL", DEFAULT_MODEL)
    if has_groq_key():
        st.caption(t(lang, "groq_ready"))
    else:
        st.warning(t(lang, "groq_missing"))

    with st.expander(t(lang, "advanced"), expanded=False):
        st.caption(t(lang, "model_info", model=chosen_model))
        extended_search = st.checkbox(t(lang, "semantic"),
                                      value=False, key="extended_search",
                                      disabled=(matrix is None or ids is None))
        st.caption(t(lang, "secret_info"))
        if st.button(t(lang, "check_groq"), disabled=not has_groq_key(), key="check_groq"):
            try:
                from groq import Groq
                groq_client = Groq(api_key=get_secret("GROQ_API_KEY"),
                                   timeout=6.0, max_retries=0)
                available = {m.id for m in groq_client.models.list().data}
                if chosen_model in available:
                    st.success(t(lang, "check_success", model=chosen_model))
                else:
                    st.error(t(lang, "check_unavailable", model=chosen_model))
            except Exception as exc:
                st.error(readable_groq_error(exc, language=lang))

    with st.form("question_form", clear_on_submit=False):
        if mode == "suggested":
            question = st.selectbox(t(lang, "question_select"), [""] + QUESTIONS[lang],
                                    format_func=lambda value: value or t(lang, "question_placeholder"),
                                    label_visibility="collapsed", key=f"question_suggested_{lang}")
        else:
            # Campo vacío por defecto: el texto de ejemplo nunca debe parecer una respuesta ya escrita.
            question = st.text_input(t(lang, "question_free"), label_visibility="visible",
                                     placeholder=t(lang, "question_free_placeholder"), max_chars=450,
                                     key=f"question_free_{lang}")
        submitted = st.form_submit_button(t(lang, "ask"), type="primary", use_container_width=True)

    if submitted and question and question.strip():
        question = question.strip()
        st.session_state.messages.append({"role": "user", "text": question, "lang": lang})
        with st.spinner(t(lang, "loading")):
            started = time.perf_counter()
            # La búsqueda permanece exactamente igual que en v5.
            retriever.embedder = None
            if extended_search and matrix is not None and ids is not None:
                try:
                    retriever.embedder = load_sentence_model()
                except Exception:
                    pass
            retrieval_ready = time.perf_counter()
            try:
                answer = answer_question(
                    question, graph, retriever,
                    api_key=get_secret("GROQ_API_KEY"),
                    model=chosen_model,
                    top_k=9,
                    response_language=lang,
                )
            except Exception as exc:
                from rag_core import Answer
                answer = Answer(
                    t(lang, "unexpected_error"), [],
                    "Erro de consulta" if lang == "pt" else "Error de consulta",
                    t(lang, "unexpected_detail", name=type(exc).__name__)
                )
            finished = time.perf_counter()
        st.session_state.messages.append({
            "role": "assistant", "text": answer.text,
            "evidence": answer.evidence, "method": answer.method,
            "notice": answer.notice,
            "elapsed": finished-started,
            "preparation": retrieval_ready-started,
            "lang": lang,
        })
        st.rerun()
    elif submitted:
        st.warning(t(lang, "choose_first"))

    if st.session_state.messages:
        st.divider()
        for message in st.session_state.messages[-20:]:
            msg_lang = message.get("lang", "es")
            if message["role"] == "user":
                st.markdown(f'<div class="msg-user"><div class="lbl">{esc(t(msg_lang,"user"))}</div>'
                            f'<div class="txt">{clean_multiline(message["text"])}</div></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="msg-bot"><div class="lbl">{esc(t(msg_lang,"assistant"))}</div>'
                            f'<div class="txt">{clean_multiline(message["text"])}</div></div>',
                            unsafe_allow_html=True)
                elapsed = message.get("elapsed")
                timing = f" · {elapsed:.1f} s" if elapsed is not None else ""
                if message.get("notice"):
                    st.warning(t(msg_lang, "groq_error"))
                with st.expander(t(lang, "evidence"), expanded=False):
                    st.caption(f'{message.get("method", "")} {timing}')
                    if message.get("notice"):
                        st.caption(message["notice"])
                    evidence_view(graph, message.get("evidence", []), lang)
        if st.button(t(lang, "clear"), use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption(t(lang, "empty_chat"))

with tab_explore:
    c1, c2 = st.columns(2)
    with c1:
        kind_label = st.selectbox(t(lang, "kind"), ["Language", "Family"],
                                  format_func=lambda value: t(lang, "languages" if value == "Language" else "families"))
    with c2:
        countries = sorted({country for e in graph.languages() for country in e.countries})
        chosen_country = st.selectbox(t(lang, "country"), [""] + countries,
                                      format_func=lambda value: country_label(lang, value) if value else t(lang, "all_countries"),
                                      disabled=(kind_label == "Family"))
    fam_options = sorted(graph.families(), key=lambda e: e.label.lower())
    fam_lookup = {f"{fam.label} · {fam.id}": fam.id for fam in fam_options}
    selected_family = st.selectbox(t(lang, "family_filter"), [""] + list(fam_lookup),
                                    format_func=lambda value: value if value else t(lang, "all_families"),
                                    disabled=(kind_label == "Family"))
    search = st.text_input(t(lang, "filter"), placeholder=t(lang, "filter_placeholder"))
    candidates = graph.languages() if kind_label == "Language" else graph.families()
    if chosen_country and kind_label == "Language":
        candidates = [e for e in candidates if chosen_country in e.countries]
    if selected_family and kind_label == "Language":
        family_id = fam_lookup[selected_family]
        candidates = [e for e in candidates if family_id in e.families]
    if search:
        from rag_core import normalize
        candidates = [e for e in candidates if normalize(search) in normalize(e.label + " " + e.iso)]
    candidates.sort(key=lambda e: e.label.casefold())
    st.caption(t(lang, "results", n=len(candidates)))
    for item in candidates[:50]:
        card(item, graph, lang)

    if candidates:
        output = io.StringIO()
        writer = csv.writer(output)
        # Las columnas de exportación se traducen; las claves RDF no se alteran.
        if lang == "pt":
            writer.writerow(["id", "tipo", "lingua_ou_familia", "iso", "paises", "falantes_aproximados", "familias"])
        else:
            writer.writerow(["id", "tipo", "lengua_o_familia", "iso", "paises", "hablantes_aproximados", "familias"])
        for item in candidates:
            kind_localized = t(lang, "type_language") if item.kind == "Language" else t(lang, "type_family")
            writer.writerow([item.id, kind_localized, item.label, item.iso,
                             "; ".join(country_label(lang, country) for country in item.countries),
                             "" if item.speakers is None else item.speakers,
                             "; ".join(item.families)])
        st.download_button(t(lang, "download"), output.getvalue().encode("utf-8-sig"),
                           file_name="selecao_linguas_rag_sa.csv" if lang == "pt" else "seleccion_lenguas_rag_sa.csv",
                           mime="text/csv", use_container_width=True)

with tab_about:
    st.markdown(f'''<div class="about-sec"><h4>{t(lang, 'about_project_title')}</h4>
<p>{t(lang, 'about_project')}</p><p>{t(lang, 'about_invite')}</p></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="about-sec"><h4>{t(lang, 'about_relations_title')}</h4>
<p>{t(lang, 'about_relations')}</p></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="about-sec"><h4>{t(lang, 'about_method_title')}</h4>
<p>{t(lang, 'about_method')}</p></div>''', unsafe_allow_html=True)
    stats = graph.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric(t(lang, "metric_languages"), stats["languages"])
    c2.metric(t(lang, "metric_families"), stats["families"])
    c3.metric(t(lang, "metric_described"), stats["with_comment"])
    st.markdown(f'''<div class="about-sec"><h4>{t(lang, 'about_technical_title')}</h4>
<p>{t(lang, 'about_technical')}</p>
<div class="chips"><span class="chip green">{t(lang, 'type_language')} → {t(lang, 'type_family')}</span>
<span class="chip green">{t(lang, 'type_language')} → {t(lang, 'country')}</span>
<span class="chip">RDF / Turtle</span><span class="chip">RDFLib</span><span class="chip">BM25</span>
<span class="chip">MiniLM (opcional)</span><span class="chip">Groq · {esc(chosen_model)}</span></div></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="about-sec"><h4>{t(lang, 'about_research_title')}</h4>
<p>{t(lang, 'about_research')}</p><p>{t(lang, 'about_note')}</p></div>''', unsafe_allow_html=True)

st.markdown(f'''<div class="tfooter"><p>🗺️ <b>{esc(t(lang,'footer'))}</b> · GraphRAG</p>
<p>{esc(t(lang,'footer_sub'))}</p></div>''', unsafe_allow_html=True)
