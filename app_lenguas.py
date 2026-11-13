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
from explore_tree import ExplorerIndex
from conversation_nav import (resolve_question, navigation_targets, focus_entity)
from relation_graph import display_local_graph
from locales import LANGUAGES, QUESTIONS, country_label, t as _locale_t

# Compatibilidad: en una actualización parcial puede seguir cargado locales.py
# de v6.4. Los nuevos textos del explorador deben funcionar igualmente.
LEGACY_TRANSLATIONS = {'es': {'explore_country_help': 'Selecciona un país para ver las agrupaciones y lenguas asociadas.',
        'explore_country_placeholder': 'Selecciona un país…',
        'explore_group': 'Agrupación principal',
        'explore_group_placeholder': 'Todas las agrupaciones del país',
        'explore_subgroup': 'Subgrupo (opcional)',
        'explore_subgroup_placeholder': 'Todos los subgrupos',
        'explore_deeper': 'Grupo más específico (opcional)',
        'explore_language': 'Lengua o variedad',
        'explore_language_placeholder': 'Selecciona una lengua…',
        'explore_search_help': 'Buscar lengua por nombre, Glottocode o ISO',
        'explore_no_matches': 'No hay lenguas asociadas a esta combinación.',
        'explore_count': '{n} lenguas o variedades en esta selección.',
        'explore_select': 'Elige una lengua para ver su ficha, o utiliza el buscador.',
        'explore_no_country': 'También puedes buscar una lengua directamente, sin elegir un país.',
        'explore_ungrouped': 'Sin agrupación registrada ({n})',
        'explore_groups_count': '{n} agrupaciones principales para este país.',
        'about_responsible_title': 'Responsable y sugerencias',
        'about_responsible': 'Responsable: Javier Maximiliano Vera Zúñiga. Puedes escribir para compartir '
                             'comentarios o sugerencias sobre el proyecto.',
        'source_links_label': 'Recursos y enlaces'},
 'pt': {'explore_country_help': 'Selecione um país para ver os agrupamentos e as línguas associados.',
        'explore_country_placeholder': 'Selecione um país…',
        'explore_group': 'Agrupamento principal',
        'explore_group_placeholder': 'Todos os agrupamentos do país',
        'explore_subgroup': 'Subgrupo (opcional)',
        'explore_subgroup_placeholder': 'Todos os subgrupos',
        'explore_deeper': 'Grupo mais específico (opcional)',
        'explore_language': 'Língua ou variedade',
        'explore_language_placeholder': 'Selecione uma língua…',
        'explore_search_help': 'Buscar língua por nome, Glottocode ou ISO',
        'explore_no_matches': 'Não há línguas associadas a esta combinação.',
        'explore_count': '{n} línguas ou variedades nesta seleção.',
        'explore_select': 'Escolha uma língua para ver sua ficha ou use a pesquisa.',
        'explore_no_country': 'Você também pode pesquisar uma língua diretamente, sem escolher um país.',
        'explore_ungrouped': 'Sem agrupamento registrado ({n})',
        'explore_groups_count': '{n} agrupamentos principais para este país.',
        'about_responsible_title': 'Responsável e sugestões',
        'about_responsible': 'Responsável: Javier Maximiliano Vera Zúñiga. Você pode escrever para compartilhar '
                             'comentários ou sugestões sobre o projeto.',
        'source_links_label': 'Recursos e links'}}

# Textos incorporados después de v6.9: permite desplegar archivos por etapas.
# No sustituye locales.py, que sigue siendo la fuente principal de traducciones.
V71_COMPAT_TRANSLATIONS = {'en': {'explore_browse_toggle': 'Browse by country and group (optional)',
        'explore_missing_country': "This entry has no country recorded in the graph's country field.",
        'explore_other_matches': 'I found a matching entry, but it is not linked to the selected country or '
                                 'group in this graph. You can still explore it here.',
        'explore_search_no_results': 'No entry matches that name or code. Try a different spelling.',
        'explore_search_placeholder': 'Kariri, kari1254, jaqaru…',
        'explore_search_scope': 'Searching the entire graph; country and family filters are not applied.',
        'explore_show_subgroups': 'Show subgroups to narrow the selection',
        'graph_alt': 'Local network of recorded relationships for {name}.',
        'graph_center': 'Center network here',
        'graph_click_help': 'Select a point in the network or use the list to open an entry.',
        'graph_explore_intro': 'Select an entry to explore its relationships in a small network.',
        'graph_help': 'Select a node to view its entry or explore further. Links represent recorded '
                      'associations, not text similarity.',
        'graph_legend': 'Green: center · Yellow: groups · Blue: languages · Gray: countries',
        'graph_limited': 'Simplified view: {n} additional links or entries are not displayed.',
        'graph_no_relations': 'This entry has no other structured links available in the graph. You can '
                              'still consult its description in the entry.',
        'graph_node_placeholder': 'Or choose an entry from the network…',
        'graph_node_select': 'Choose a node (mobile-friendly option)',
        'graph_open': 'Open entry in Explore',
        'graph_selected': 'Selected: {name}',
        'graph_title': 'Relationships of {name}'},
 'es': {'explore_browse_toggle': 'Explorar por país y agrupación (opcional)',
        'explore_missing_country': 'Esta ficha no tiene un país registrado en el campo de países del grafo.',
        'explore_other_matches': 'Encontré la ficha que buscas, pero no figura entre las relaciones del '
                                 'país o agrupación seleccionados. Puedes consultarla aquí sin cambiar los '
                                 'datos del grafo.',
        'explore_search_no_results': 'No encontramos una ficha con ese nombre o código. Puedes probar otra '
                                     'escritura.',
        'explore_search_placeholder': 'Kariri, kari1254, jaqaru…',
        'explore_search_scope': 'Búsqueda en todo el grafo, sin filtros de país o familia.',
        'explore_show_subgroups': 'Ver subgrupos para precisar la búsqueda',
        'graph_alt': 'Red local de relaciones documentadas para {name}.',
        'graph_center': 'Centrar red aquí',
        'graph_click_help': 'Selecciona un punto de la red o utiliza la lista para abrir una ficha.',
        'graph_explore_intro': 'Al seleccionar una ficha, también podrás explorar sus relaciones en una red '
                               'pequeña.',
        'graph_help': 'Selecciona un nodo para consultar su ficha o continuar explorando. Las conexiones '
                      'muestran asociaciones registradas, no semejanzas entre textos.',
        'graph_legend': 'Verde: ficha central · Amarillo: agrupaciones · Azul: lenguas · Gris: países',
        'graph_limited': 'Vista simplificada: hay {n} relaciones o fichas adicionales fuera de esta vista.',
        'graph_no_relations': 'Esta ficha no tiene otras relaciones estructuradas disponibles en el grafo. '
                              'Puedes consultar su descripción en la ficha.',
        'graph_node_placeholder': 'O selecciona una ficha de la red…',
        'graph_node_select': 'Elegir un nodo (opción para celular)',
        'graph_open': 'Abrir ficha en Explorar',
        'graph_selected': 'Seleccionaste: {name}',
        'graph_title': 'Relaciones de {name}'},
 'pt': {'explore_browse_toggle': 'Explorar por país e agrupamento (opcional)',
        'explore_missing_country': 'Este registro não tem um país informado no campo de países do grafo.',
        'explore_other_matches': 'Encontrei o registro que você procura, mas ele não aparece entre as '
                                 'relações do país ou agrupamento selecionado. Você pode consultá-lo aqui '
                                 'sem alterar os dados do grafo.',
        'explore_search_no_results': 'Não encontramos um registro com esse nome ou código. Tente outra '
                                     'grafia.',
        'explore_search_placeholder': 'Kariri, kari1254, jaqaru…',
        'explore_search_scope': 'Pesquisa em todo o grafo, sem filtros de país ou família.',
        'explore_show_subgroups': 'Ver subgrupos para refinar a pesquisa',
        'graph_alt': 'Rede local de relações registradas para {name}.',
        'graph_center': 'Centralizar a rede aqui',
        'graph_click_help': 'Selecione um ponto da rede ou use a lista para abrir um registro.',
        'graph_explore_intro': 'Ao selecionar um registro, você também poderá explorar suas relações em uma '
                               'pequena rede.',
        'graph_help': 'Selecione um nó para consultar sua ficha ou continuar explorando. As conexões '
                      'mostram associações registradas, não semelhanças entre textos.',
        'graph_legend': 'Verde: registro central · Amarelo: agrupamentos · Azul: línguas · Cinza: países',
        'graph_limited': 'Visualização simplificada: há {n} relações ou registros adicionais fora desta '
                         'rede.',
        'graph_no_relations': 'Este registro não tem outras relações estruturadas disponíveis no grafo. '
                              'Você pode consultar sua descrição na ficha.',
        'graph_node_placeholder': 'Ou selecione um registro da rede…',
        'graph_node_select': 'Escolher um nó (opção para celular)',
        'graph_open': 'Abrir ficha em Explorar',
        'graph_selected': 'Você selecionou: {name}',
        'graph_title': 'Relações de {name}'}}


def t(language: str, key: str, **kwargs) -> str:
    """Resuelve textos con prioridad para locales.py y respaldo de compatibilidad.

    La búsqueda de un idioma alternativo no reemplaza una traducción propia.
    Nunca se imprimen nombres de claves en el sitio público.
    """
    try:
        return _locale_t(language, key, **kwargs)
    except KeyError as exc:
        # Respaldo para archivos de idioma de v6.9 y v7.0 en la nube.
        template = (V71_COMPAT_TRANSLATIONS.get(language, {}).get(key)
                    or LEGACY_TRANSLATIONS.get(language, {}).get(key)
                    or V71_COMPAT_TRANSLATIONS['es'].get(key)
                    or LEGACY_TRANSLATIONS['es'].get(key))
        if template is None:
            raise KeyError(
                f"Falta el texto '{key}'. Actualiza locales.py junto con app_lenguas.py."
            ) from exc
        return template.format(**kwargs)


ROOT = Path(__file__).resolve().parent
TTL = ROOT / "grafo_enriquecido.ttl"
MATRIX = ROOT / "embed_matrix_v2.npy"
ID_MAP = ROOT / "id_map_v2.pkl"

st.set_page_config(page_title="Lenguas indígenas / Línguas indígenas / Indigenous languages", page_icon="🗺️",
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
        if e.kind == "Language":
            metadata.append(f"Glottocode {e.id}")
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
    metadata = [f"Glottocode {e.id}"] if e.kind == "Language" else []
    if e.iso:
        metadata.append(f"ISO {e.iso}")
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


# Selector trilingüe. Nombres de lenguas y datos RDF se mantienen sin cambios.
lang = st.session_state.get("lang", "es")
st.markdown(f'''<div class="app-header"><div class="app-header-title">🗺️ {esc(t(lang, 'header'))}</div>
<div class="app-header-sub">{esc(t(lang, 'subtitle'))}</div></div>''', unsafe_allow_html=True)

intro_col, language_col = st.columns([4, 1.3], vertical_alignment="center")
with intro_col:
    st.markdown(f'<p class="intro">{esc(t(lang, "intro"))}</p>', unsafe_allow_html=True)
with language_col:
    selected_lang = st.selectbox("🌐 Idioma / Language", list(LANGUAGES),
                                 format_func=lambda value: LANGUAGES[value],
                                 index=list(LANGUAGES).index(lang), key="lang_choice")
if selected_lang != lang:
    # Las pestañas tienen etiquetas distintas al cambiar de idioma.
    previous_tabs = [t(lang, k) for k in ("tab_chat", "tab_explore", "tab_about")]
    next_tabs = [t(selected_lang, k) for k in ("tab_chat", "tab_explore", "tab_about")]
    current_tab = st.session_state.get("main_tabs", previous_tabs[0])
    st.session_state["main_tabs"] = (next_tabs[previous_tabs.index(current_tab)]
                                      if current_tab in previous_tabs else next_tabs[0])
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

# Pestañas controlables: permiten navegar desde una respuesta a «Explorar».
# on_change y key requieren Streamlit >= 1.55.
tab_chat, tab_explore, tab_about = st.tabs(
    [t(lang, "tab_chat"), t(lang, "tab_explore"), t(lang, "tab_about")],
    on_change="rerun", key="main_tabs")

if "messages" not in st.session_state:
    st.session_state.messages = []


def open_in_explorer(kind: str, value: str):
    """Acción de navegación, ejecutada antes del siguiente rerun."""
    destination = focus_entity(graph, kind, value)
    if destination is None:
        return
    st.session_state["nav_focus"] = {"kind": kind, "value": value}
    # Evita que filtros anteriores oculten el destino.
    for widget_key in list(st.session_state.keys()):
        if widget_key.startswith(("explore_group_", "explore_sub_", "explore_deep_", "explore_language_")):
            del st.session_state[widget_key]
    if kind == "country":
        st.session_state["explore_country"] = value
        st.session_state["explore_search"] = ""
    elif kind == "language":
        # La ficha y su grafo se muestran arriba; no duplicarlos abajo.
        st.session_state["explore_country"] = ""
        st.session_state["explore_search"] = ""
    else:
        st.session_state["explore_country"] = ""
        st.session_state["explore_search"] = ""
    st.session_state["main_tabs"] = t(st.session_state.get("lang", "es"), "tab_explore")


def close_explorer_focus():
    st.session_state.pop("nav_focus", None)

with tab_chat:
    st.markdown(f"### {t(lang, 'chat_section_title')}")
    st.caption(t(lang, "chat_section_help"))
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

    # Ambas vías están disponibles a la vez: no es necesario seleccionar un modo
    # ni pulsar un círculo. Si se escribe una pregunta, esa tiene prioridad.
    with st.form("question_form", clear_on_submit=False):
        suggested_question = st.selectbox(
            t(lang, "question_select"), [""] + QUESTIONS[lang],
            format_func=lambda value: value or t(lang, "question_placeholder"),
            key=f"question_suggested_{lang}",
        )
        st.caption(t(lang, "question_or_write"))
        free_question = st.text_input(
            t(lang, "question_free"),
            placeholder=t(lang, "question_free_placeholder"), max_chars=450,
            key=f"question_free_{lang}",
        )
        submitted = st.form_submit_button(t(lang, "ask"), type="primary", use_container_width=True)

    question = (free_question or "").strip() or (suggested_question or "").strip()
    if submitted and question:
        question = question.strip()
        resolved = resolve_question(question, st.session_state.messages, graph)
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
                    resolved.for_retrieval, graph, retriever,
                    api_key=get_secret("GROQ_API_KEY"),
                    model=chosen_model,
                    top_k=9,
                    response_language=lang,
                )
            except Exception as exc:
                from rag_core import Answer
                answer = Answer(
                    t(lang, "unexpected_error"), [],
                    ("Query error" if lang == "en" else "Erro de consulta" if lang == "pt" else "Error de consulta"),
                    t(lang, "unexpected_detail", name=type(exc).__name__)
                )
            finished = time.perf_counter()
        st.session_state.messages.append({
            "role": "assistant", "text": answer.text,
            "evidence": answer.evidence, "method": answer.method,
            "notice": answer.notice,
            "elapsed": finished-started,
            "preparation": retrieval_ready-started,
            "subject_ids": answer.subject_ids,
            "country_filter": answer.country_filter,
            "family_filter": answer.family_filter,
            "continued": resolved.continued,
            "lang": lang,
        })
        st.rerun()
    elif submitted:
        st.warning(t(lang, "choose_first"))

    if st.session_state.messages:
        st.divider()
        for message_index, message in enumerate(st.session_state.messages[-20:]):
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
                if message.get("continued") and message.get("subject_ids"):
                    subject = graph.entities.get(message["subject_ids"][0])
                    if subject:
                        st.caption(t(msg_lang, "continued_topic", name=subject.label))
                if message.get("notice"):
                    st.warning(t(msg_lang, "groq_error"))
                if not message.get("notice"):
                    actions = navigation_targets(
                        graph, subject_ids=message.get("subject_ids", []),
                        evidence_ids=message.get("evidence", []),
                        country=message.get("country_filter"),
                        family=message.get("family_filter"))
                    if actions:
                        action_columns = st.columns(len(actions))
                        for col, action in zip(action_columns, actions):
                            with col:
                                title = t(lang, "nav_" + action.kind,
                                          name=(country_label(lang, action.name)
                                                if action.kind == "country" else action.name))
                                st.button(title, key=f"nav_{message_index}_{action.kind}_{action.value}",
                                          use_container_width=True,
                                          on_click=open_in_explorer,
                                          args=(action.kind, action.value))
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
    st.markdown(t(lang, "explore_intro"))
    st.caption(t(lang, "graph_explore_intro"))
    explorer = ExplorerIndex.from_graph(graph)
    # Buscar tiene prioridad también sobre una ficha abierta antes desde el chat:
    # así nunca aparecen dos fichas de lenguas diferentes al mismo tiempo.
    search = st.text_input(
        t(lang, "explore_search_help"),
        placeholder=t(lang, "explore_search_placeholder"),
        key="explore_search",
    )
    result_ids: list[str] = []
    if search.strip():
        st.session_state.pop("nav_focus", None)
    focus = st.session_state.get("nav_focus")
    if focus:
        focus_kind, focus_value = focus.get("kind", ""), focus.get("value", "")
        focused = focus_entity(graph, focus_kind, focus_value)
        if focused is not None:
            if focus_kind == "country":
                st.markdown(f"#### {esc(country_label(lang, focus_value))}")
                st.caption(t(lang, "nav_country_count", n=len(focused)))
            else:
                st.caption(t(lang, "nav_from_answer"))
                card(focused, graph, lang)
                # v7.0: vista local de las relaciones RDF, sin alterar las fichas.
                display_local_graph(graph, focused.kind.lower(), focused.id, lang,
                                    t, country_label, open_in_explorer,
                                    scope="focused", expanded=True)
                if focus_kind == "language":
                    related = navigation_targets(graph, subject_ids=[focused.id])[1:]
                    if related:
                        columns = st.columns(len(related))
                        for col, target in zip(columns, related):
                            with col:
                                st.button(t(lang, "nav_" + target.kind,
                                            name=(country_label(lang, target.name)
                                                  if target.kind == "country" else target.name)),
                                          key=f"focus_related_{target.kind}_{target.value}",
                                          on_click=open_in_explorer,
                                          args=(target.kind, target.value), use_container_width=True)
                elif focus_kind == "family" and focused.members:
                    valid_members = [mid for mid in focused.members if mid in graph.entities]
                    if valid_members:
                        chosen_member = st.selectbox(
                            t(lang, "nav_choose_member"), [""] + valid_members,
                            format_func=lambda mid: (graph.entities[mid].label if mid
                                                     else t(lang, "explore_language_placeholder")),
                            key=f"focus_family_member_{focused.id}")
                        if chosen_member in valid_members:
                            card(graph.entities[chosen_member], graph, lang)
            st.button(t(lang, "nav_close"), key="nav_close_focus", on_click=close_explorer_focus)
            st.divider()

    if search.strip():
        result_ids = explorer.language_ids("", query=search)
        if not result_ids:
            st.info(t(lang, "explore_search_no_results"))
        else:
            st.caption(t(lang, "explore_search_scope"))
            st.caption(t(lang, "explore_count", n=len(result_ids)))
            # Una coincidencia única se abre sin pasar por otro desplegable.
            if len(result_ids) == 1:
                selected_language = result_ids[0]
            else:
                selected_language = st.selectbox(
                    t(lang, "explore_language"), [""] + result_ids[:50],
                    format_func=lambda eid: (
                        f"{explorer.languages[eid].label} · {eid}"
                        if eid else t(lang, "explore_language_placeholder")
                    ),
                    key=f"explore_search_result_{lang}",
                )
            if selected_language in result_ids:
                entry = explorer.languages[selected_language]
                card(entry, graph, lang)
                if not entry.countries:
                    st.caption(t(lang, "explore_missing_country"))
                display_local_graph(graph, "language", selected_language, lang,
                                    t, country_label, open_in_explorer,
                                    scope="direct_search", expanded=True)
    else:
        # Navegación optativa para quien prefiere explorar los registros del país.
        # No es necesario escoger familia para llegar a una lengua.
        stored_country = st.session_state.get("explore_country", "")
        with st.expander(t(lang, "explore_browse_toggle"),
                         expanded=bool(stored_country)):
            country_names = sorted(explorer.countries,
                                   key=lambda c: normalize(country_label(lang, c)))
            chosen_country = st.selectbox(
                t(lang, "country"), [""] + country_names,
                format_func=lambda c: (
                    country_label(lang, c) if c
                    else t(lang, "explore_country_placeholder")
                ),
                key="explore_country",
            )
            active_group = ""
            ungrouped = False
            if chosen_country:
                main_ids = explorer.main_groups(chosen_country)
                without_group = explorer.ungrouped_ids(chosen_country)
                main_options = [""] + main_ids + (["__ungrouped__"] if without_group else [])
                selected_main = st.selectbox(
                    t(lang, "explore_group"), main_options,
                    format_func=lambda fid: (
                        t(lang, "explore_group_placeholder") if not fid
                        else t(lang, "explore_ungrouped", n=len(without_group))
                        if fid == "__ungrouped__"
                        else explorer.label(fid, chosen_country)
                    ),
                    key=f"explore_group_{chosen_country}",
                )
                ungrouped = selected_main == "__ungrouped__"
                if selected_main and not ungrouped:
                    active_group = selected_main
                    children = explorer.child_groups(selected_main, chosen_country)
                    if children:
                        detail = st.checkbox(t(lang, "explore_show_subgroups"),
                                             value=False,
                                             key=f"explore_more_{chosen_country}_{selected_main}")
                        if detail:
                            selected_sub = st.selectbox(
                                t(lang, "explore_subgroup"), [""] + children,
                                format_func=lambda fid: (
                                    explorer.label(fid, chosen_country)
                                    if fid else t(lang, "explore_subgroup_placeholder")
                                ),
                                key=f"explore_sub_{chosen_country}_{selected_main}",
                            )
                            active_group = selected_sub or selected_main
        if not chosen_country:
            st.caption(t(lang, "explore_no_country"))
        else:
            if ungrouped:
                result_ids = explorer.ungrouped_ids(chosen_country)
            else:
                result_ids = explorer.language_ids(chosen_country, active_group)
            st.caption(t(lang, "explore_count", n=len(result_ids)))
            if not result_ids:
                st.info(t(lang, "explore_no_matches"))
            else:
                selected_language = st.selectbox(
                    t(lang, "explore_language"), [""] + result_ids,
                    format_func=lambda eid: (
                        explorer.languages[eid].label if eid
                        else t(lang, "explore_language_placeholder")
                    ),
                    key=f"explore_language_{chosen_country}_{active_group}_{ungrouped}_{lang}",
                )
                if selected_language in result_ids:
                    entry = explorer.languages[selected_language]
                    card(entry, graph, lang)
                    same_focus = (focus and focus.get("kind") == "language"
                                  and focus.get("value") == selected_language)
                    if not same_focus:
                        display_local_graph(graph, "language", selected_language, lang,
                                            t, country_label, open_in_explorer,
                                            scope="country_browse", expanded=True)
                else:
                    st.caption(t(lang, "explore_select"))

    # Exportar exactamente el conjunto mostrado; sin modificar datos RDF.
    if result_ids:
        output = io.StringIO()
        writer = csv.writer(output)
        if lang == "pt":
            writer.writerow(["glottocode", "tipo", "lingua_ou_familia", "iso", "paises", "falantes_aproximados", "familias"])
        elif lang == "en":
            writer.writerow(["glottocode", "type", "language_or_family", "iso", "countries", "approximate_speakers", "families"])
        else:
            writer.writerow(["glottocode", "tipo", "lengua_o_familia", "iso", "paises", "hablantes_aproximados", "familias"])
        for eid in result_ids:
            item = explorer.languages[eid]
            writer.writerow([item.id, t(lang, "type_language"), item.label, item.iso,
                             "; ".join(country_label(lang, c) for c in item.countries),
                             "" if item.speakers is None else item.speakers,
                             "; ".join(item.families)])
        st.download_button(t(lang, "download"), output.getvalue().encode("utf-8-sig"),
                           file_name=("selecao_linguas_rag_sa.csv" if lang == "pt" else
                                      "language_selection_rag_sa.csv" if lang == "en" else
                                      "seleccion_lenguas_rag_sa.csv"),
                           mime="text/csv", use_container_width=True)

with tab_about:
    st.markdown(f'''<div class="about-sec"><h4>{t(lang, 'about_project_title')}</h4>
<p>{t(lang, 'about_project')}</p></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="about-sec"><h4>{t(lang, 'about_method_title')}</h4>
<p>{t(lang, 'about_method')}</p></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="about-sec"><h4>{t(lang, 'about_sources_title')}</h4>
<p>{t(lang, 'about_sources')}</p></div>''', unsafe_allow_html=True)
    source_links = [
        ("Glottolog", "https://glottolog.org/"),
        ("Wikidata", "https://www.wikidata.org/"),
        ("Wikipedia", "https://www.wikipedia.org/"),
        ("Ethnologue", "https://www.ethnologue.com/"),
    ]
    links_html = " · ".join(
        f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(label)}</a>'
        for label, url in source_links
    )
    st.markdown(f'<div class="about-sec"><h4>{esc(t(lang, "source_links_label"))}</h4>'
                f'<p>{links_html}</p></div>', unsafe_allow_html=True)
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
    st.markdown(f'''<div class="about-sec"><h4>{t(lang, 'about_scope_title')}</h4>
<p>{t(lang, 'about_scope')}</p></div>''', unsafe_allow_html=True)

    st.markdown(f'<div class="about-sec"><h4>{esc(t(lang, "about_responsible_title"))}</h4>'
                f'<p>{esc(t(lang, "about_responsible"))} '
                f'<a href="mailto:jxvera@gmail.com">jxvera@gmail.com</a></p></div>',
                unsafe_allow_html=True)

st.markdown(f'''<div class="tfooter"><p>🗺️ <b>{esc(t(lang,'footer'))}</b> · GraphRAG</p></div>''', unsafe_allow_html=True)
