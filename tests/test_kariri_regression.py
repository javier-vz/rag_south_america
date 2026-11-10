"""Regresiones con entidades y datos copiados de la versión pública del TTL.

El caso reproduce el fallo de la captura: 'lengua' también es una familia
real (ex:leng1262), pero la entidad solicitada es Kariri (ex:kari1254).
"""
from pathlib import Path

import pytest

from rag_core import (KnowledgeGraph, Retriever, answer_question, build_evidence_context,
                      detect_family, match_language_names, structured_answer)


SNAPSHOT = '''
@prefix ex: <http://example.org/lang/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:leng1262 a ex:Family ; ex:label "Lengua" .
ex:quec1387 a ex:Family ; ex:label "Quechuan" .
ex:lengA a ex:Language ; ex:label "Enlhet Norte" ;
  ex:wikidata_countries "Paraguay" ; ex:belongsToFamily ex:leng1262 .
ex:lengB a ex:Language ; ex:label "Enxet Sur" ;
  ex:wikidata_countries "Paraguay" ; ex:belongsToFamily ex:leng1262 .
ex:kari1254 a ex:Language ; ex:label "Kariri" ;
  ex:latitude -7.434628e+00 ; ex:longitude -4.048727e+01 ;
  ex:wikipedia_summary "The Karirí languages, generally considered dialects of a single language, are a group of languages formerly spoken by the Kiriri people of Brazil. It was spoken until the middle of the 20th century; the 4,000 ethnic Kiriri are now monolingual Portuguese speakers, though a few know common phrases and names of medicinal plants. A revival of the Dzubukuá variety has been ongoing since 1989." ;
  rdfs:comment "Kariri is a language." .
ex:ayac1239 a ex:Language ; ex:label "Ayacucho Quechua" ;
  ex:wikidata_countries "Perú" ; ex:belongsToFamily ex:quec1387 .
'''


@pytest.fixture
def graph(tmp_path):
    filename = tmp_path / "snapshot.ttl"
    filename.write_text(SNAPSHOT, encoding="utf-8")
    return KnowledgeGraph.from_turtle(filename)


@pytest.mark.parametrize("question", [
    "Qué sabes sobre la lengua kariri?",
    "¿Qué sabes de Karirí?",
    "Háblame sobre Kariri",
    "¿A qué familia pertenece la lengua Kariri?",
    "¿En qué país se hablaba Kariri?",
])
def test_kariri_prioritized_not_lengua_family(graph, question):
    r = Retriever(graph).search(question)
    assert r.ids[0] == "kari1254"
    assert "lengA" not in r.ids
    assert "lengB" not in r.ids
    assert r.subject_ids == ["kari1254"]
    assert r.family_filter is None


def test_plain_word_lengua_does_not_activate_family(graph):
    assert detect_family("¿Qué sabes sobre la lengua Kariri?", graph) is None
    assert detect_family("¿A qué familia pertenece la lengua Kariri?", graph) is None
    assert detect_family("¿Cuáles son las lenguas de la familia Lengua?", graph) == "leng1262"
    result = Retriever(graph).search("¿Cuáles son las lenguas de la familia Lengua?")
    assert {"lengA", "lengB"}.issubset(set(result.ids))


def test_exact_named_context_contains_real_kariri_description(graph):
    question = "Qué sabes sobre la lengua kariri?"
    r = Retriever(graph).search(question)
    facts = structured_answer(question, graph, r)
    assert facts.evidence == ["kari1254"]
    assert "Kariri" in facts.text
    import rag_core
    context, ids = rag_core.combine_grounding(question, graph, r, facts)
    assert ids == ["kari1254"]
    assert "Kiriri people of Brazil" in context
    assert "Dzubukuá" in context
    assert "Enlhet" not in context
    assert "Enxet" not in context


def test_groq_gets_kariri_and_no_wrong_language(graph, monkeypatch):
    import rag_core
    received = []
    def fake(q, context, api_key, model):
        received.append(context)
        return ("Kariri fue hablada históricamente por el pueblo Kiriri en Brasil. "
                "La variedad dzubukuá ha tenido iniciativas de revitalización.")
    monkeypatch.setattr(rag_core, "generate_groq", fake)
    answer = answer_question("¿Qué sabes sobre la lengua kariri?", graph, Retriever(graph), api_key="prueba")
    assert len(received) == 1
    assert "Kariri" in received[0]
    assert "Enlhet" not in received[0]
    assert "Kariri" in answer.text
    assert answer.method.endswith("Groq")


def test_unknown_name_does_not_fall_back_to_lengua_family(graph):
    question = "¿Qué sabes sobre la lengua zaraquiri?"
    r = Retriever(graph).search(question)
    assert r.ids == []
    assert r.unknown_subject is True
    answer = structured_answer(question, graph, r)
    assert "No hay una ficha" in answer.text


def test_country_and_other_language_still_work(graph):
    peru = Retriever(graph).search("¿Qué lenguas están en Perú?")
    assert peru.country_filter == "Perú"
    assert "ayac1239" in peru.ids
    q = "¿Qué sabes del Ayacucho Quechua?"
    r = Retriever(graph).search(q)
    assert r.ids[0] == "ayac1239"
    assert match_language_names(q, graph) == ["ayac1239"]
