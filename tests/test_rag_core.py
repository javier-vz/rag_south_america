import pickle
from pathlib import Path

import numpy as np
import pytest

from rag_core import (KnowledgeGraph, Retriever, answer_question, build_evidence_context,
                      detect_country, detect_family, load_vectors, safe_url, structured_answer,
                      useful_country_examples, SYSTEM_PROMPT)

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
    assert kg.entities["spanish1"].needs_indigenous_review
    assert not kg.entities["alive1"].needs_indigenous_review
    assert "spanish1" in kg.entities
    assert "spanish1" not in {e.id for e in useful_country_examples(kg.country("Perú"), "Perú")}


def test_structured_country_count_not_misrepresented_as_census(kg):
    q = "¿Cuántas lenguas indígenas se hablan actualmente en Perú?"
    a = structured_answer(q, kg, Retriever(kg).search(q))
    assert "3 fichas" in a.text  # 4 del país, una marca no indígena explícita
    assert "no equivale a un censo" in a.text


def test_qualitative_query_no_unrequested_speaker_counts(kg):
    q = "¿Qué lenguas indígenas se hablan en Perú?"
    r = Retriever(kg).search(q)
    a = structured_answer(q, kg, r)
    assert "Lengua indígena A" in a.text
    assert "120" not in a.text


def test_embedded_text_and_generated_comments_are_distinguished(kg):
    context = build_evidence_context(kg, ["alive1", "unknown1"])
    assert "generada por IA" in context
    assert "Wikipedia" in context
    assert "Estimación de hablantes" not in context


def test_country_family_detection(kg):
    assert detect_country("Lenguas del Perú") == "Perú"
    assert detect_family("¿Qué hay en la familia Quechua?", kg) == "quec1387"
    assert detect_family("familia tupí-guaraní", kg) == "tupi1276"


def test_family_explanation_not_forced_list(kg):
    q = "¿Qué sabemos de la familia Quechua?"
    r = Retriever(kg).search(q)
    assert structured_answer(q, kg, r) is None
    assert "quec1387" in r.ids


def test_amazonia_not_inferred_from_country(kg):
    q = "¿Qué lenguas se hablan en la Amazonía peruana?"
    a = structured_answer(q, kg, Retriever(kg).search(q))
    assert "no tiene una clasificación comprobada" in a.text


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
    assert "Perú" in a.text


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
    assert "APROXIMACIONES" in SYSTEM_PROMPT


def test_auditor_generates_csv_and_json(kg, tmp_path):
    from auditar_grafo import audit
    p = tmp_path / "sample.ttl"
    p.write_text(SAMPLE_TTL, encoding="utf-8")
    report, csv_path, json_path = audit(p, tmp_path / "out")
    assert report["language_records"] == 5
    assert report["source_links_embedded_in_local_uris"] == 1
    assert report["peru_records"] == 4
    assert csv_path.is_file() and json_path.is_file()
