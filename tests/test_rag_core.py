import pickle
from pathlib import Path

import numpy as np
import pytest

from rag_core import (KnowledgeGraph, Retriever, Entity, answer_question, build_evidence_context,
                      detect_country, detect_family, load_vectors, safe_url, structured_answer,
                      useful_country_examples, is_obviously_non_indigenous, SYSTEM_PROMPT)

SAMPLE_TTL = '''
@prefix ex: <http://example.org/lang/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:quec1387 a ex:Family ; ex:label "Quechuan" .
ex:tupi1276 a ex:Family ; ex:label "Tupi-Guarani" .
ex:alive1 a ex:Language ; ex:label "Lengua indígena A" ; ex:iso "aaa" ;
  ex:belongsToFamily ex:quec1387 ; ex:wikidata_countries "Perú" ;
  ex:wikidata_speakers "120" ;
  ex:describedByResource <http://example.org/lang/https://www.wikidata.org/entity/Q1> ;
  rdfs:comment "Comentario IA de prueba" .
ex:unknown1 a ex:Language ; ex:label "Lengua sin cifra" ;
  ex:belongsToFamily ex:quec1387 ; ex:wikidata_countries "Perú" ;
  ex:wikipedia_summary "Texto documental recuperado de muestra" .
ex:zero1 a ex:Language ; ex:label "Lengua con cero" ;
  ex:wikidata_countries "Perú" ; ex:wikidata_speakers "0" .
ex:spanish1 a ex:Language ; ex:label "Peruvian Amazonian Spanish" ;
  ex:wikidata_countries "Perú" .
ex:alive2 a ex:Language ; ex:label "Lengua indígena B" ;
  ex:wikidata_countries "Bolivia" ; ex:wikidata_speakers "55" .
'''


@pytest.fixture
def kg(tmp_path):
    f = tmp_path / "sample.ttl"
    f.write_text(SAMPLE_TTL, encoding="utf-8")
    return KnowledgeGraph.from_turtle(f)


def test_rdf_parsed_and_source_links_normalized(kg):
    assert kg.stats()["languages"] == 5
    assert kg.entities["alive1"].sources == ["https://www.wikidata.org/entity/Q1"]
    assert "alive1" in kg.entities["quec1387"].members


def test_missing_is_not_zero_or_extinction(kg):
    assert kg.entities["unknown1"].speakers is None
    assert kg.entities["zero1"].speakers == 0
    assert kg.stats()["without_speakers"] == 2
    assert kg.stats()["with_zero_speakers"] == 1


def test_indigeneity_warning_is_non_destructive(kg):
    assert is_obviously_non_indigenous(kg.entities["spanish1"])
    assert not is_obviously_non_indigenous(kg.entities["alive1"])
    assert "spanish1" in kg.entities
    assert "spanish1" not in {e.id for e in useful_country_examples(kg.country("Perú"), "Perú")}


def test_structured_country_count_not_misrepresented_as_census(kg):
    q = "¿Cuántas lenguas indígenas se hablan actualmente en Perú?"
    a = structured_answer(q, kg, Retriever(kg).search(q))
    assert "3 lenguas o variedades" in a.text  # 4 del país, una variedad claramente no indígena
    assert "no es un censo completo" in a.text


def test_qualitative_query_no_unrequested_speaker_counts(kg):
    q = "¿Qué lenguas indígenas se hablan en Perú?"
    r = Retriever(kg).search(q)
    a = structured_answer(q, kg, r)
    assert "Lengua indígena A" in a.text
    assert "120" not in a.text


def test_simple_relations_in_context_no_unrequested_speaker_counts(kg):
    context = build_evidence_context(kg, ["alive1", "unknown1"])
    assert "País(es): Perú" in context
    assert "Pertenece a: Quechuan" in context
    assert "120" not in context
    assert "Lengua sin cifra" in context


def test_country_family_detection(kg):
    assert detect_country("Lenguas del Perú") == "Perú"
    assert detect_family("¿Qué hay en la familia Quechua?", kg) == "quec1387"
    assert detect_family("familia tupí-guaraní", kg) == "tupi1276"


def test_family_membership_comes_from_rdf(kg):
    q = "¿Qué sabemos de la familia Quechua?"
    r = Retriever(kg).search(q)
    factual = structured_answer(q, kg, r)
    assert "familia Quechuan" in factual.text
    assert "Lengua indígena A" in factual.text
    assert "quec1387" in r.ids


def test_amazonia_not_inferred_from_country(kg):
    q = "¿Qué lenguas se hablan en la Amazonía peruana?"
    a = structured_answer(q, kg, Retriever(kg).search(q))
    assert "no hay información suficiente" in a.text


@pytest.mark.parametrize("q", [
    "¿Cuántas lenguas hay en Perú?",
    "¿Qué lenguas indígenas se hablan en Perú?",
    "¿Qué sabemos de la familia Quechua?",
    "Háblame de la lengua indígena A",
])
def test_groq_used_for_all_questions(kg, monkeypatch, q):
    import rag_core
    calls = []
    def fake(question, context, api_key, model):
        calls.append((question, context, api_key, model))
        return "En el Perú conviven muchas lenguas y tradiciones, como la Lengua indígena A."
    monkeypatch.setattr(rag_core, "generate_groq", fake)
    result = answer_question(q, kg, Retriever(kg), api_key="clave-de-prueba")
    assert len(calls) == 1
    assert result.method.endswith("Groq")
    assert "En el Perú" in result.text


def test_groq_failure_explicit_and_not_silent(kg, monkeypatch):
    import rag_core
    def unavailable(*args):
        raise TimeoutError()
    monkeypatch.setattr(rag_core, "generate_groq", unavailable)
    a = answer_question("¿Qué lenguas indígenas hay en Perú?", kg, Retriever(kg), api_key="clave")
    assert "Groq" in a.notice
    assert "Groq no disponible" in a.method
    assert "no pude conectar" in a.text
    assert "Lengua indígena" not in a.text  # no simular una respuesta del LLM


def test_no_key_disclosed(kg):
    a = answer_question("¿Qué lenguas hay en Perú?", kg, Retriever(kg), api_key="")
    assert "GROQ_API_KEY" in a.notice
    assert "sin LLM" not in a.text


def test_arbitrary_citation_is_removed_not_shown(kg, monkeypatch):
    import rag_core
    monkeypatch.setattr(rag_core, "generate_groq", lambda *args: "Hay registros relacionados. [G999]")
    result = answer_question("¿Qué lenguas hay en Perú?", kg, Retriever(kg), api_key="clave")
    assert "G999" not in result.text


def test_vectors_validate_dimensions_and_normalize(tmp_path):
    p = tmp_path / "vectors.npy"
    np.save(p, np.array([[2.,0.], [0.,3.]], dtype=np.float32))
    im = tmp_path / "id.pkl"
    with im.open("wb") as f:
        pickle.dump(["alive1","alive2"], f)
    a, ids = load_vectors(p, im)
    assert np.allclose(np.linalg.norm(a,axis=1),1)
    assert ids == ["alive1", "alive2"]
    with im.open("wb") as f:
        pickle.dump(["alive1"], f)
    with pytest.raises(ValueError):
        load_vectors(p, im)


def test_url_sanitizes_local_external_and_blocks_javascript():
    assert safe_url("http://example.org/lang/https://en.wikipedia.org/wiki/Quechua") == "https://en.wikipedia.org/wiki/Quechua"
    assert safe_url("javascript:alert(1)") == ""


def test_prompt_asks_for_friendly_uncertainty(kg):
    assert "No lo sé con seguridad" in SYSTEM_PROMPT
    assert "PÁRRAFO breve" in SYSTEM_PROMPT
    assert "muy aproximadas" in SYSTEM_PROMPT


def test_no_attributeerror_if_cached_entity_lacks_old_property(kg, monkeypatch):
    """Regresión de la captura: property ausente tras una actualización de Streamlit."""
    monkeypatch.delattr(Entity, "needs_indigenous_review")
    q = "¿Qué lenguas indígenas se hablan actualmente en Perú?"
    ret = Retriever(kg).search(q)
    assert "spanish1" not in ret.ids
    factual = structured_answer(q, kg, ret)
    assert "Lengua indígena A" in factual.text
    assert kg.stats()["for_review"] == 1


def test_simple_relations_and_similar_text_not_kinship(kg):
    kg.entities["alive1"].similar = ["alive2"]
    assert "alive2" not in {x.id for x in kg.family("quec1387")}
    q = "¿Qué lenguas pertenecen a la familia quechua?"
    r = Retriever(kg).search(q)
    assert "alive1" in r.ids
    assert "alive2" not in r.ids


def test_country_question_facts_dont_force_speaker_counts(kg):
    q = "¿Qué lenguas se hablan en Perú?"
    ret = Retriever(kg).search(q)
    facts = structured_answer(q, kg, ret)
    assert "120" not in facts.text
    assert "Lengua sin cifra" in facts.text


def test_no_key_does_not_pretend_llm_answer(kg):
    answer = answer_question("¿Qué lenguas se hablan en Perú?", kg, Retriever(kg), api_key="")
    assert "Groq no disponible" == answer.method
    assert "GROQ_API_KEY" in answer.notice
    assert "Lengua indígena A" not in answer.text


def test_pocos_hablantes_prioritizes_lowest_observed_number(kg):
    result = structured_answer("¿Cuáles tienen pocos hablantes?", kg,
                               Retriever(kg).search("¿Cuáles tienen pocos hablantes?"))
    assert result is not None
    assert "Lengua indígena B" in result.text
    assert result.text.index("Lengua indígena B") < result.text.index("Lengua indígena A")
