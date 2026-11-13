"""Regresiones de UX: búsqueda global, ruta opcional por país y traducciones."""
from pathlib import Path

from explore_tree import ExplorerIndex
from locales import t
from rag_core import KnowledgeGraph

ROOT = Path(__file__).resolve().parents[1]


def sample_explorer(tmp_path):
    ttl = tmp_path / "graph.ttl"
    ttl.write_text('''@prefix ex: <http://example.org/lang/> .
ex:kari1254 a ex:Language ; ex:label "Kariri" ;
    ex:wikipedia_summary "Kariri was historically spoken in Brazil" .
ex:asha1243 a ex:Language ; ex:label "Asháninka" ; ex:iso "cni" ;
    ex:wikidata_countries "Perú" ; ex:belongsToFamily ex:araw1281 .
ex:araw1281 a ex:Family ; ex:label "Arawakan" .
''', encoding="utf-8")
    return ExplorerIndex.from_graph(KnowledgeGraph.from_turtle(ttl))


def test_direct_search_can_find_entity_without_country(tmp_path):
    index = sample_explorer(tmp_path)
    assert index.language_ids("", query="kariri") == ["kari1254"]
    assert index.language_ids("", query="kari1254") == ["kari1254"]
    assert not index.language_ids("Perú", query="kariri")


def test_country_browse_does_not_require_family(tmp_path):
    index = sample_explorer(tmp_path)
    assert index.language_ids("Perú") == ["asha1243"]
    assert index.language_ids("Perú", "araw1281") == ["asha1243"]
    assert index.language_ids("", query="cni") == ["asha1243"]


def test_ui_direct_search_has_priority_over_previous_focus():
    source = (ROOT / "app_lenguas.py").read_text(encoding="utf-8")
    index_search = source.index('search = st.text_input(', source.index('with tab_explore:'))
    index_focus = source.index('focus = st.session_state.get("nav_focus")', index_search)
    index_country = source.index('with st.expander(t(lang, "explore_browse_toggle")')
    assert index_search < index_focus < index_country
    assert 'st.session_state.pop("nav_focus", None)' in source
    assert 'result_ids = explorer.language_ids("", query=search)' in source
    assert 'if len(result_ids) == 1:' in source


def test_all_navigation_texts_in_three_languages():
    keys = ["explore_browse_toggle", "explore_search_help", "explore_search_scope",
            "explore_missing_country", "explore_search_no_results",
            "explore_show_subgroups", "explore_search_placeholder"]
    for language in ("es", "pt", "en"):
        for key in keys:
            assert t(language, key)
        assert "Glottocode" in t(language, "explore_search_help")
