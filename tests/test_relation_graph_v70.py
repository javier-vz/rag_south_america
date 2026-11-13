"""Regresión v7.0: redes locales pequeñas, legibles y sin enlaces inventados.

No requiere API Groq, Streamlit en ejecución ni el grafo completo externo.
"""
from pathlib import Path

import pytest

from rag_core import KnowledgeGraph
from relation_graph import (GraphEdge, GraphNode, LocalGraph, build_figure,
                            build_local_graph, graph_targets, node_coordinates,
                            node_key, selected_node)
from locales import LANGUAGES, STRINGS, t

RDF_SAMPLE = '''@prefix ex: <http://example.org/lang/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:kari1254 a ex:Language ; ex:label "Kariri" ;
  ex:wikipedia_summary "Kariri was historically spoken by the Kiriri people of Brazil." ;
  ex:semanticallySimilarTo ex:jaqa1244 .
ex:ayma1253 a ex:Family ; ex:label "Aymaran" .
ex:quec1387 a ex:Family ; ex:label "Quechuan" .
ex:jaqa1244 a ex:Language ; ex:label "Jaqaru" ;
  ex:belongsToFamily ex:ayma1253 ; ex:wikidata_countries "Perú" ;
  ex:semanticallySimilarTo ex:kari1254 .
ex:sout2996 a ex:Language ; ex:label "Southern Aymara" ;
  ex:belongsToFamily ex:ayma1253 ; ex:wikidata_countries "Perú" .
ex:ayac1239 a ex:Language ; ex:label "Ayacucho Quechua" ;
  ex:belongsToFamily ex:quec1387 ; ex:wikidata_countries "Perú" .
'''


@pytest.fixture
def graph(tmp_path):
    file = tmp_path / 'tiny.ttl'
    file.write_text(RDF_SAMPLE, encoding='utf-8')
    return KnowledgeGraph.from_turtle(file)


def test_jaqaru_has_only_actual_structured_relationships(graph):
    view = build_local_graph(graph, 'language', 'jaqa1244')
    assert view.center == 'language:jaqa1244'
    assert {'language:jaqa1244', 'family:ayma1253', 'country:Perú',
            'language:sout2996'} == {n.key for n in view.nodes}
    assert { (e.start, e.end, e.kind) for e in view.edges } == {
        ('language:jaqa1244', 'family:ayma1253', 'belongs_to'),
        ('language:jaqa1244', 'country:Perú', 'country'),
        ('family:ayma1253', 'language:sout2996', 'shares_family'),
    }


def test_semantic_similarity_never_becomes_relationship(graph):
    view = build_local_graph(graph, 'language', 'kari1254')
    assert len(view.nodes) == 1
    assert not view.edges
    assert all(n.value != 'jaqa1244' for n in view.nodes)


def test_no_cross_family_connections(graph):
    view = build_local_graph(graph, 'family', 'ayma1253')
    assert {'language:jaqa1244', 'language:sout2996', 'family:ayma1253'} == {
        n.key for n in view.nodes
    }
    assert not any(n.value == 'ayac1239' for n in view.nodes)
    assert len(view.edges) == 2


def test_country_is_not_guessed_from_descriptions(graph):
    view = build_local_graph(graph, 'language', 'kari1254')
    assert not any(n.kind == 'country' for n in view.nodes)


def test_invalid_target_is_rejected(graph):
    with pytest.raises(ValueError):
        build_local_graph(graph, 'language', 'fake')
    with pytest.raises(ValueError):
        build_local_graph(graph, 'country', 'Perú')
    with pytest.raises(ValueError):
        build_local_graph(graph, 'family', 'kari1254')


def test_graph_target_maps_to_existing_destinations(graph):
    view = build_local_graph(graph, 'language', 'jaqa1244')
    targets = graph_targets(view)
    assert targets['country:Perú'] == ('country', 'Perú')
    assert targets['family:ayma1253'] == ('family', 'ayma1253')
    assert targets['language:sout2996'] == ('language', 'sout2996')


def test_selected_node_accepts_plotly_selection_event(graph):
    view = build_local_graph(graph, 'language', 'jaqa1244')
    event = {'selection': {'points': [{'customdata': ['family:ayma1253']}], 'box': [], 'lasso': []}}
    selected = selected_node(event, view)
    assert selected is not None
    assert (selected.kind, selected.value) == ('family', 'ayma1253')


@pytest.mark.parametrize('state', [None, {}, {'selection': {'points': []}},
                                      {'selection': {'points': [{'customdata': ['language:missing']}]}},
                                      {'selection': {'points': [{'customdata': []}]}}])
def test_invalid_selection_is_ignored(graph, state):
    view = build_local_graph(graph, 'language', 'jaqa1244')
    assert selected_node(state, view) is None


def test_selection_skips_untrusted_points(graph):
    view = build_local_graph(graph, 'language', 'jaqa1244')
    event = {'selection': {'points': [
        {'customdata': ['family:ayma1253']},
        {'customdata': ['family:forged']},
    ]}}
    assert selected_node(event, view).value == 'ayma1253'


def test_node_positions_are_stable_and_unique(graph):
    view = build_local_graph(graph, 'language', 'jaqa1244')
    points = node_coordinates(view)
    assert set(points) == {n.key for n in view.nodes}
    assert len(set(points.values())) == len(points)
    assert points[view.center] == (0.0, 0.0)


def test_visualization_is_selectable_and_dark(graph):
    view = build_local_graph(graph, 'language', 'jaqa1244')
    fig = build_figure(view, 'es')
    assert fig.layout.clickmode == 'event+select'
    assert fig.layout.paper_bgcolor == '#0d1117'
    assert len(fig.data) > len(view.edges)
    assert any('language:jaqa1244' in (list(trace.customdata)[0] or [])
               for trace in fig.data if trace.customdata is not None)


def test_big_family_is_capped_with_information(graph):
    from rag_core import Entity
    for i in range(30):
        eid = f'x{i:02d}'
        graph.entities[eid] = Entity(id=eid, kind='Language', label=f'Example {i:02d}',
                                     families=['ayma1253'])
        graph.entities['ayma1253'].members.append(eid)
    view = build_local_graph(graph, 'family', 'ayma1253', max_members=8)
    assert len(view.nodes) == 9
    assert view.omitted == 24
    assert len(view.edges) == 8


def test_language_graph_caps_members_in_large_family(graph):
    from rag_core import Entity
    for i in range(25):
        eid = f'x{i:02d}'
        graph.entities[eid] = Entity(id=eid, kind='Language', label=f'Example {i:02d}',
                                     families=['ayma1253'])
        graph.entities['ayma1253'].members.append(eid)
    view = build_local_graph(graph, 'language', 'jaqa1244', max_related=5)
    related = [n for n in view.nodes if n.kind == 'language' and n.value != 'jaqa1244']
    assert len(related) == 5
    assert view.omitted >= 20


def test_no_mutation_of_original_rdf_entities(graph):
    before = {id_: (list(e.families), list(e.members), list(e.countries))
              for id_,e in graph.entities.items()}
    build_local_graph(graph, 'language', 'jaqa1244')
    build_local_graph(graph, 'family', 'ayma1253')
    after = {id_: (list(e.families), list(e.members), list(e.countries))
             for id_,e in graph.entities.items()}
    assert before == after


def test_labels_and_graph_module_are_trilingual():
    assert set(LANGUAGES) == {'es', 'pt', 'en'}
    assert set(STRINGS['es']) == set(STRINGS['pt']) == set(STRINGS['en'])
    for lang in LANGUAGES:
        assert 'Jaqaru' in t(lang, 'graph_title', name='Jaqaru')
        assert 'Jaqaru' in t(lang, 'graph_alt', name='Jaqaru')
        assert '3' in t(lang, 'graph_limited', n=3)
        assert t(lang, 'graph_click_help')


def test_chart_has_accessible_text_fallback_and_navigation_wiring():
    app = (Path(__file__).resolve().parents[1]/'app_lenguas.py').read_text('utf-8')
    relation = (Path(__file__).resolve().parents[1]/'relation_graph.py').read_text('utf-8')
    req = (Path(__file__).resolve().parents[1]/'requirements.txt').read_text('utf-8')
    assert app.count('display_local_graph(') >= 2
    assert 'on_select="rerun"' in relation
    assert 'st.selectbox(' in relation
    assert 'on_click=open_in_explorer' in relation
    assert 'plotly>=6,<7' in req


def test_language_only_graph_still_renders(graph):
    view = build_local_graph(graph, 'language', 'kari1254')
    fig = build_figure(view, 'pt')
    assert len(fig.data) == 1
    assert fig.data[0].customdata[0][0] == 'language:kari1254'
