"""Busca por nombre/Glottocode/ISO y prueba el caso Kariri sin país RDF."""
from pathlib import Path
from explore_tree import ExplorerIndex
from rag_core import KnowledgeGraph, match_language_names, Retriever

SAMPLE = """@prefix ex: <http://example.org/lang/> .
ex:kari1254 a ex:Language ; ex:label "Kariri" ;
 ex:wikipedia_summary "The Karirí languages were formerly spoken in Brazil." .
ex:asha1243 a ex:Language ; ex:label "Asháninka" ;
 ex:iso "cni" ; ex:wikidata_countries "Perú" .
ex:abip1241 a ex:Language ; ex:label "Abipon" ;
 ex:wikidata_countries "Argentina" .
"""

def make_graph(tmp_path):
    ttl=tmp_path/'sample.ttl'
    ttl.write_text(SAMPLE, encoding='utf-8')
    return KnowledgeGraph.from_turtle(ttl)

def test_glottocode_search_without_country(tmp_path):
    explorer=ExplorerIndex.from_graph(make_graph(tmp_path))
    assert explorer.language_ids('', query='kari1254') == ['kari1254']
    assert explorer.language_ids('', query='kariri') == ['kari1254']
    assert explorer.language_ids('', query='cni') == ['asha1243']

def test_country_does_not_invent_a_relation(tmp_path):
    explorer=ExplorerIndex.from_graph(make_graph(tmp_path))
    assert explorer.language_ids('Brasil', query='Kariri') == []
    assert explorer.language_ids('Perú', query='Kariri') == []
    assert explorer.language_ids('', query='Kariri') == ['kari1254']
    assert explorer.languages['kari1254'].countries == []

def test_direct_search_by_glottocode_in_chat(tmp_path):
    g=make_graph(tmp_path)
    assert match_language_names('¿Qué sabes de kari1254?',g)==['kari1254']
    assert Retriever(g).search('¿Qué sabes sobre la lengua kari1254?').ids[0]=='kari1254'

def test_glottocode_search_after_country_filter(tmp_path):
    explorer=ExplorerIndex.from_graph(make_graph(tmp_path))
    assert explorer.language_ids('Perú',query='cni') == ['asha1243']
    assert explorer.language_ids('Perú',query='abip1241') == []
    assert explorer.language_ids('',query='abip1241') == ['abip1241']

def test_ui_copy_translated_and_uses_glottocode():
    from locales import t
    for lang in ('es','pt','en'):
        assert 'Glottocode' in t(lang,'explore_search_help')
        assert 'Glottocode' in t(lang,'explore_intro')
        assert t(lang, 'explore_other_matches')

def test_app_searches_globally_before_optional_country_filters():
    s=(Path(__file__).resolve().parents[1]/'app_lenguas.py').read_text()
    assert 'result_ids = explorer.language_ids("", query=search)' in s
    assert 'with st.expander(t(lang, "explore_browse_toggle")' in s
    assert 'if len(result_ids) == 1:' in s
    assert 'Glottocode {e.id}' in s
