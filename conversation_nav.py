"""Continuidad conversacional y navegación contextual para RAG South America.

Este módulo no modifica el RDF, no infiere parentescos a partir de similitud y
no llama al LLM: resuelve referencias breves y ofrece destinos documentados.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag_core import (KnowledgeGraph, detect_country, detect_family,
                      match_language_names, normalize,
                      requested_unknown_language)


@dataclass(frozen=True)
class ResolvedQuestion:
    original: str
    for_retrieval: str
    subject_ids: tuple[str, ...] = ()
    continued: bool = False


@dataclass(frozen=True)
class NavigationTarget:
    kind: str  # language, family, country
    value: str  # RDF entity ID, or exact country value in RDF
    name: str


# Una pregunta corta es seguimiento cuando contiene una referencia contextual.
# No tratamos todas las preguntas como continuaciones por el simple hecho de
# tener un historial: así se evita asociar consultas nuevas con la lengua anterior.
_FOLLOWUP_PREFIX = re.compile(
    r"^(?:y\b|e\b|and\b|but\b|pero\b|mas\b|what about\b|"
    r"y que\b|e o que\b|and what\b)", re.I)
_ANAPHORA = re.compile(
    r"\b(?:ella|ello|esa lengua|esta lengua|la misma|la anterior|su familia|"
    r"sus variedades|de ella|sobre ella|dela|dessa lingua|essa lingua|"
    r"a mesma|sua familia|seus dialetos|it|its|that language|this language|"
    r"the same language|its family|its speakers|them)\b", re.I)
_ELLIPTICAL_INTENTS = (
    "donde", "onde", "where", "hablaba", "fala", "spoken", "speak", "vive", "vivia",
    "familia", "family", "subgrupo", "subgroup", "pais", "paises", "country", "countries",
    "origen", "origin",
    "recuper", "revitaliz", "revival", "revive", "revital", "todavia", "ainda",
    "still", "actualmente", "nowadays", "hoy", "hoje", "today", "extinta", "extinct",
    "siguen", "continu", "se conoce", "know about", "sabemos", "sabe", "known",
)


def _looks_like_followup(question: str) -> bool:
    q = normalize(question)
    if _FOLLOWUP_PREFIX.search(q) or _ANAPHORA.search(q):
        return True
    # «¿Dónde se hablaba?» / «Where was it spoken?» / «Onde era falada?»
    # Solo por elipsis si no se mencionó explícitamente otra entidad.
    words = q.split()
    return len(words) <= 11 and any(phrase in q for phrase in _ELLIPTICAL_INTENTS)


def _last_topic(messages: list[dict[str, Any]], graph: KnowledgeGraph) -> list[str]:
    # No saltamos un último turno sin referente: cambiar de tema debe cortar
    # la memoria, aunque en turnos antiguos hubiera otra lengua.
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        ids = message.get("subject_ids") or []
        return [eid for eid in ids
                if eid in graph.entities and graph.entities[eid].kind == "Language"][:2]
    return []


def resolve_question(question: str, messages: list[dict], graph: KnowledgeGraph) -> ResolvedQuestion:
    text = question.strip()
    direct_subjects = match_language_names(text, graph)
    if direct_subjects:
        return ResolvedQuestion(text, text, tuple(direct_subjects))

    # Una lengua solicitada explícitamente pero ausente no debe convertirse
    # silenciosamente en la lengua de la conversación previa.
    if requested_unknown_language(text, []):
        return ResolvedQuestion(text, text)

    topic = _last_topic(messages, graph)
    if not topic or not _looks_like_followup(text):
        return ResolvedQuestion(text, text)

    # «¿Qué lenguas hay en Brasil?» es otra búsqueda, no otra pregunta sobre
    # Kariri. Un país explícito se mantiene solo si hay anáfora clara («¿y
    # ella también se habla en Brasil?»).
    explicit_country = bool(detect_country(text))
    explicit_family = bool(detect_family(text, graph))
    q = normalize(text)
    if (explicit_country or explicit_family) and not (
        _ANAPHORA.search(q) or _FOLLOWUP_PREFIX.search(q)
    ):
        return ResolvedQuestion(text, text)
    name = graph.entities[topic[0]].label
    # Información de resolución, no una afirmación lingüística. Las
    # relaciones y descripciones siguen saliendo exclusivamente del grafo.
    resolved = f"{text}\n\n[Referencia de esta pregunta: lengua {name}.]"
    return ResolvedQuestion(text, resolved, tuple(topic), True)


def navigation_targets(graph: KnowledgeGraph, *, subject_ids=(),
                       evidence_ids=(), country=None, family=None) -> list[NavigationTarget]:
    """Acciones de navegación basadas solo en relaciones RDF existentes."""
    actions: list[NavigationTarget] = []

    def add(kind, value, label):
        if value and not any(a.kind == kind and a.value == value for a in actions):
            actions.append(NavigationTarget(kind, str(value), str(label)))

    main = next((graph.entities[eid] for eid in subject_ids
                 if eid in graph.entities and graph.entities[eid].kind == "Language"), None)
    if main is None and not country and not family and len(evidence_ids) == 1:
        main = next((graph.entities[eid] for eid in evidence_ids
                     if eid in graph.entities and graph.entities[eid].kind == "Language"), None)
    if main:
        add("language", main.id, main.label)
        valid_families = [graph.entities[fid] for fid in main.families
                          if fid in graph.entities and graph.entities[fid].kind == "Family"]
        if valid_families:
            # La agrupación más amplia registrada se elige como atajo inicial,
            # sin convertir las relaciones semánticas en parentescos.
            fam = max(valid_families, key=lambda x: (len(x.members), x.label))
            add("family", fam.id, fam.label)
        for linked_country in main.countries[:1]:
            add("country", linked_country, linked_country)
    else:
        if family and family in graph.entities and graph.entities[family].kind == "Family":
            add("family", family, graph.entities[family].label)
        if country:
            add("country", country, country)
    return actions[:3]


def focus_entity(graph: KnowledgeGraph, kind: str, value: str):
    """Devuelve la ficha RDF o el conjunto país asociado, o None."""
    if kind in ("language", "family"):
        entity = graph.entities.get(value)
        return entity if entity and entity.kind.lower() == kind else None
    if kind == "country":
        return graph.country(value) if value in {c for e in graph.languages() for c in e.countries} else None
    return None
