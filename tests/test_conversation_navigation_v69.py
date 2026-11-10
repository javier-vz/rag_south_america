"""Regresión v6.8/v6.9: seguimiento contextual y navegación real.

No requiere API de Groq ni un archivo RDF externo.
"""
from pathlib import Path
import re

import pytest

from conversation_nav import (resolve_question, navigation_targets, focus_entity,
                              NavigationTarget)
from rag_core import KnowledgeGraph, Retriever, answer_question, structured_answer
from locales import LANGUAGES, STRINGS, country_label, t

GRAPH = '''@prefix ex: <http://example.org/lang/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:kari1254 a ex:Language ; ex:label "Kariri" ;
  ex:wikipedia_summary "Kariri was historically spoken by the Kiriri people in Brazil. Revitalization of Dzubukuá has been ongoing." .
ex:ayma1253 a ex:Family ; ex:label "Aymaran" .
ex:quec1387 a ex:Family ; ex:label "Quechuan" .
ex:jaqa1244 a ex:Language ; ex:label "Jaqaru" ;
  ex:belongsToFamily ex:ayma1253 ; ex:wikidata_countries "Perú" .
ex:sout2996 a ex:Language ; ex:label "Southern Aymara" ;
  ex:belongsToFamily ex:ayma1253 ; ex:wikidata_countries "Perú" .
ex:ayac1239 a ex:Language ; ex:label "Ayacucho Quechua" ;
  ex:belongsToFamily ex:quec1387 ; ex:wikidata_countries "Perú" .
ex:bras1234 a ex:Language ; ex:label "Example language" ;
  ex:wikidata_countries "Brasil" .
'''


@pytest.fixture
def graph(tmp_path):
    path = tmp_path / 'small.ttl'
    path.write_text(GRAPH, encoding='utf-8')
    return KnowledgeGraph.from_turtle(path)


def last_answer(subjects, country=None, family=None):
    return [{"role": "user", "text": "Earlier question"},
            {"role": "assistant", "text": "Earlier answer", "subject_ids": subjects,
             "country_filter": country, "family_filter": family}]


@pytest.mark.parametrize("language,first,followup", [
    ('es', '¿Qué sabes sobre la lengua Kariri?', '¿Y dónde se hablaba?'),
    ('pt', 'O que você sabe sobre a língua Kariri?', 'E onde era falada?'),
    ('en', 'What do you know about Kariri?', 'And where was it spoken?'),
    ('es', '¿Qué sabes de Kariri?', '¿Se está intentando recuperar?'),
    ('pt', 'Conte sobre Kariri.', 'A revitalização continua?'),
    ('en', 'Tell me about Kariri.', 'Is it being revived?'),
])
def test_followup_keeps_kariri(graph, language, first, followup):
    first_result = resolve_question(first, [], graph)
    assert first_result.subject_ids == ('kari1254',)
    memory = last_answer(list(first_result.subject_ids))
    result = resolve_question(followup, memory, graph)
    assert result.continued is True
    assert result.subject_ids == ('kari1254',)
    assert 'Kariri' in result.for_retrieval
    assert Retriever(graph).search(result.for_retrieval).subject_ids == ['kari1254']


@pytest.mark.parametrize('question', [
    '¿Qué sabes sobre Jaqaru?',
    'O que você sabe sobre Jaqaru?',
    'What do you know about Jaqaru?',
])
def test_new_named_language_changes_topic(graph, question):
    result = resolve_question(question, last_answer(['kari1254']), graph)
    assert not result.continued
    assert result.subject_ids == ('jaqa1244',)
    assert 'Kariri' not in result.for_retrieval


@pytest.mark.parametrize('question', [
    '¿Qué lenguas se hablan en Perú?',
    'Quais línguas são faladas no Peru?',
    'Which languages are spoken in Peru?',
])
def test_new_country_question_does_not_inherit_kariri(graph, question):
    result = resolve_question(question, last_answer(['kari1254']), graph)
    assert not result.continued
    assert 'Kariri' not in result.for_retrieval


def test_no_memory_after_country_turn_or_clear(graph):
    country_answer = last_answer([] , country='Perú')
    result = resolve_question('¿Y dónde se hablaba?', country_answer, graph)
    assert not result.continued
    assert not resolve_question('¿Y dónde se hablaba?', [], graph).continued


def test_unknown_named_subject_does_not_inherit_last(graph):
    result = resolve_question('¿Qué sabes de la lengua zaraquiri?',
                              last_answer(['kari1254']), graph)
    assert not result.continued
    assert 'Kariri' not in result.for_retrieval


def test_followup_with_pronoun_and_country_retains_subject(graph):
    result = resolve_question('¿Y ella se habla en Perú?', last_answer(['kari1254']),graph)
    assert result.continued
    assert 'Kariri' in result.for_retrieval


def test_focus_kariri_has_no_invented_country_or_family(graph):
    results = navigation_targets(graph, subject_ids=['kari1254'], evidence_ids=['kari1254'])
    assert results == [NavigationTarget('language', 'kari1254', 'Kariri')]
    assert focus_entity(graph, 'language', 'kari1254').label == 'Kariri'


def test_jaqaru_links_real_family_country(graph):
    results = navigation_targets(graph, subject_ids=['jaqa1244'], evidence_ids=['jaqa1244'])
    assert [(x.kind,x.value) for x in results] == [
        ('language', 'jaqa1244'), ('family','ayma1253'), ('country','Perú')]
    assert len(focus_entity(graph,'family','ayma1253').members) == 2
    assert len(focus_entity(graph,'country','Perú')) == 3


def test_country_and_family_question_links(graph):
    c = navigation_targets(graph, country='Perú', evidence_ids=['jaqa1244','ayac1239'])
    f = navigation_targets(graph, family='ayma1253', evidence_ids=['jaqa1244'])
    assert [(x.kind, x.value) for x in c] == [('country','Perú')]
    assert [(x.kind, x.value) for x in f] == [('family','ayma1253')]


def test_invalid_focus_is_not_opened(graph):
    assert focus_entity(graph,'family','kari1254') is None
    assert focus_entity(graph,'country','Atlantis') is None
    assert focus_entity(graph,'language','fake') is None


def test_groq_receives_resolved_entity_and_returns_metadata(graph,monkeypatch):
    import rag_core
    captures=[]
    def fake_generate(question, context, api_key, model):
        captures.append((question,context))
        return 'Kariri was historically spoken in Brazil.'
    monkeypatch.setattr(rag_core,'generate_groq',fake_generate)
    resolved=resolve_question('And where was it spoken?',last_answer(['kari1254']),graph)
    answer=answer_question(resolved.for_retrieval,graph,Retriever(graph),api_key='test',response_language='en')
    assert answer.subject_ids==['kari1254']
    assert answer.method.endswith('Groq')
    assert len(captures)==1
    assert 'Kariri' in captures[0][1] and 'Brazil' in captures[0][1]


def test_navigation_text_is_trilingual_and_template_safe():
    assert list(LANGUAGES)==['es','pt','en']
    assert set(STRINGS['es'])==set(STRINGS['pt'])==set(STRINGS['en'])
    for lang in LANGUAGES:
        assert 'Jaqaru' in t(lang,'nav_language',name='Jaqaru')
        assert country_label(lang,'Perú') in t(lang,'nav_country',name=country_label(lang,'Perú'))
        assert 'Kariri' in t(lang,'continued_topic',name='Kariri')


def test_streamlit_tab_navigation_and_callback_are_in_source():
    app=(Path(__file__).resolve().parents[1]/'app_lenguas.py').read_text('utf-8')
    assert 'on_change="rerun", key="main_tabs"' in app
    assert 'on_click=open_in_explorer' in app
    assert 'st.session_state["main_tabs"] = t(' in app
    assert 'resolved.for_retrieval, graph, retriever' in app
    assert '"subject_ids": answer.subject_ids' in app
    requirements=(Path(__file__).resolve().parents[1]/'requirements.txt').read_text('utf-8')
    assert 'streamlit>=1.55' in requirements


def test_followup_elliptical_country_question(graph):
    result = resolve_question('¿En qué país se hablaba?', last_answer(['kari1254']), graph)
    assert result.continued is True
    assert 'Kariri' in result.for_retrieval


def test_no_arbitrary_link_from_many_evidence_records(graph):
    nav = navigation_targets(graph, evidence_ids=['kari1254', 'jaqa1244'])
    assert nav == []
