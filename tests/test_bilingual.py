"""La capa bilingüe no cambia las relaciones ni impide respuestas con Groq."""
import pytest
from locales import LANGUAGES, QUESTIONS, STRINGS, country_label, t
from rag_core import KnowledgeGraph, Retriever, answer_question, count_intent, structured_answer

SAMPLE = '''
@prefix ex: <http://example.org/lang/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:kari1254 a ex:Language ; ex:label "Kariri" ;
  ex:wikipedia_summary "Kariri was spoken by the Kiriri people in Brazil." .
ex:quec1387 a ex:Family ; ex:label "Quechuan" .
ex:ayac1239 a ex:Language ; ex:label "Ayacucho Quechua" ;
  ex:belongsToFamily ex:quec1387 ; ex:wikidata_countries "Perú" .
'''

@pytest.fixture
def graph(tmp_path):
    p = tmp_path / "sample.ttl"
    p.write_text(SAMPLE, encoding="utf-8")
    return KnowledgeGraph.from_turtle(p)


def test_all_ui_strings_have_both_languages():
    assert list(LANGUAGES) == ["es", "pt"]
    assert set(STRINGS["es"]) == set(STRINGS["pt"])
    assert all(STRINGS[lang][key] for lang in LANGUAGES for key in STRINGS[lang])
    assert "Português" == LANGUAGES["pt"]
    assert country_label("pt", "Perú") == "Peru"
    assert t("pt", "tab_about") == "ℹ️ Sobre o projeto"


def test_kariri_is_in_both_question_lists():
    assert any("kariri" in text.lower() for text in QUESTIONS["es"])
    assert any("kariri" in text.lower() for text in QUESTIONS["pt"])


@pytest.mark.parametrize("question", [
    "¿Qué sabes sobre la lengua kariri?",
    "O que você sabe sobre a língua kariri?",
])
def test_kariri_retrieved_in_es_and_pt(graph, question):
    result = Retriever(graph).search(question)
    assert result.subject_ids == ["kari1254"]
    assert result.ids[0] == "kari1254"


def test_portuguese_country_question_uses_same_graph(graph):
    q = "Quais línguas indígenas são faladas atualmente no Peru?"
    retrieved = Retriever(graph).search(q)
    assert retrieved.country_filter == "Perú"
    assert "ayac1239" in retrieved.ids
    facts = structured_answer(q, graph, retrieved)
    assert "Ayacucho Quechua" in facts.text


def test_portuguese_counts_recognized(graph):
    q = "Quantas línguas estão associadas ao Peru?"
    assert count_intent(q)
    facts = structured_answer(q, graph, Retriever(graph).search(q))
    assert "1 lenguas o variedades" in facts.text


def test_groq_receives_explicit_locale_without_changing_question(graph, monkeypatch):
    import rag_core
    got = []
    def fake(question, context, api_key, model):
        got.append((question, context))
        return "Kariri é uma língua histórica do povo Kiriri no Brasil."
    monkeypatch.setattr(rag_core, "generate_groq", fake)
    question = "¿Qué sabes sobre la lengua kariri?"
    result = answer_question(question, graph, Retriever(graph),
                             api_key="prueba", response_language="pt")
    assert result.text.startswith("Kariri é")
    assert got[0][0] == question
    assert "IDIOMA_DE_RESPUESTA: português brasileiro" in got[0][1]
    assert "Kariri" in got[0][1]


def test_portuguese_no_key_message_is_friendly(graph):
    answer = answer_question("O que você sabe sobre a língua kariri?", graph,
                             Retriever(graph), api_key="", response_language="pt")
    assert "Não consigo responder" in answer.text
    assert "segredos do Streamlit" in answer.notice
