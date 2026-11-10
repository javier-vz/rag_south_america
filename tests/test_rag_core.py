import pickle
from pathlib import Path

import numpy as np
import pytest

from rag_core import (KnowledgeGraph, Retriever, answer_question, build_evidence_context,
                      detect_country, detect_family, load_vectors, safe_url, structured_answer)

SAMPLE_TTL = '''
@prefix ex: <http://example.org/lang/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:quec1387 a ex:Family ; ex:label "Quechuan" .
ex:tupi1276 a ex:Family ; ex:label "Tupi-Guarani" .
ex:alive1 a ex:Language ; ex:label "Lengua viva A" ; ex:iso "aaa" ;
  ex:belongsToFamily ex:quec1387 ; ex:wikidata_countries "Perú" ;
  ex:wikidata_speakers "120" ;
  ex:describedByResource <http://example.org/lang/https://www.wikidata.org/entity/Q1> ;
  rdfs:comment "Descripción para prueba" .
ex:unknown1 a ex:Language ; ex:label "Lengua sin cifra" ;
  ex:belongsToFamily ex:quec1387 ; ex:wikidata_countries "Perú" ;
  ex:wikipedia_summary "Sin información de hablantes" .
ex:zero1 a ex:Language ; ex:label "Lengua con cero" ;
  ex:wikidata_countries "Perú" ; ex:wikidata_speakers "0" .
ex:alive2 a ex:Language ; ex:label "Lengua viva B" ;
  ex:wikidata_countries "Bolivia" ; ex:wikidata_speakers "55" .
'''


@pytest.fixture
def kg(tmp_path):
    ttl = tmp_path / "sample.ttl"
    ttl.write_text(SAMPLE_TTL, encoding="utf-8")
    return KnowledgeGraph.from_turtle(ttl)


def test_parsing_relationships_and_sources(kg):
    assert kg.stats() == {
        "languages": 4, "families": 2, "with_comment": 1,
        "with_speakers": 3, "without_speakers": 1, "with_zero_speakers": 1,
    }
    assert "alive1" in kg.entities["quec1387"].members
    assert kg.entities["alive1"].sources == ["https://www.wikidata.org/entity/Q1"]
    assert kg.entities["unknown1"].speaker_status == "sin dato"
    assert kg.entities["zero1"].speaker_status == "0 registrado"
    assert kg.entities["alive1"].iso == "aaa"


def test_country_count_is_not_a_census(kg):
    r = Retriever(kg).search("¿Cuántas lenguas existen actualmente en Perú?")
    a = structured_answer("¿Cuántas lenguas existen actualmente en Perú?", kg, r)
    assert "3 registros" in a.text
    assert "1 no tienen cifra" in a.text
    assert "1 tienen 0 registrado" in a.text
    assert "no equivale" in a.text


def test_current_query_does_not_call_unknown_extinct(kg):
    q = "¿Qué lenguas se hablan actualmente en Perú?"
    answer = answer_question(q, kg, Retriever(kg))
    assert "Lengua viva A" in answer.text
    assert "sin fecha" in answer.text
    assert "Lengua con cero" not in answer.text
    assert "Lengua sin cifra" not in answer.text
    assert "extinta" not in answer.text.lower()


def test_family_filter(kg):
    q = "¿Qué lenguas pertenecen a la familia Quechua?"
    result = Retriever(kg).search(q)
    assert detect_family(q, kg) == "quec1387"
    assert set(result.ids) == {"alive1", "unknown1"}
    answer = answer_question(q, kg, Retriever(kg))
    assert "Quechuan" in answer.text


def test_longest_family_alias_wins(kg):
    assert detect_family("Familia tupí-guaraní", kg) == "tupi1276"
    assert detect_country("Lenguas del Perú") == "Perú"


def test_bm25_offline_and_evidence(kg):
    result = Retriever(kg).search("Lengua viva A")
    assert result.ids[0] == "alive1"
    context = build_evidence_context(kg, ["alive1"])
    assert "[G1]" in context
    assert "(sin fecha verificada)" in context
    assert "requiere verificación" in context


def test_missing_key_gives_graceful_fallback(kg):
    result = answer_question("Descríbeme la lengua viva A", kg, Retriever(kg), api_key="")
    assert result.method == "Recuperación de respaldo (sin LLM)"
    assert result.evidence
    assert "[G1]" in result.text


def test_vectors_validation(tmp_path):
    matrix = np.array([[2., 0.], [0., 3.]], dtype=np.float32)
    np.save(tmp_path / "matrix.npy", matrix)
    with (tmp_path / "id_map.pkl").open("wb") as f:
        pickle.dump(["alive1", "alive2"], f)
    arr, ids = load_vectors(tmp_path / "matrix.npy", tmp_path / "id_map.pkl")
    assert ids == ["alive1", "alive2"]
    assert np.allclose(np.linalg.norm(arr, axis=1), 1)
    with (tmp_path / "id_map.pkl").open("wb") as f:
        pickle.dump(["alive1"], f)
    with pytest.raises(ValueError, match="Desajuste"):
        load_vectors(tmp_path / "matrix.npy", tmp_path / "id_map.pkl")


def test_safe_url_rejects_script_scheme():
    assert safe_url("javascript:alert(1)") == ""
    assert safe_url("http://example.org/lang/https://en.wikipedia.org/wiki/Quechua") == "https://en.wikipedia.org/wiki/Quechua"


def test_multilingual_and_no_invalid_risk(kg):
    result = answer_question("¿Qué lenguas están en peligro crítico?", kg, Retriever(kg))
    assert "no incluye una clasificación de vitalidad validada" in result.text


def test_explanatory_question_uses_retrieval_not_fake_list(kg):
    q = "¿Qué sabemos de la familia Quechua?"
    result = Retriever(kg).search(q)
    assert "quec1387" in result.ids
    assert structured_answer(q, kg, result) is None


def test_amazonia_is_not_the_same_as_country(kg):
    q = "¿Qué lenguas se hablan en la Amazonía peruana?"
    result = answer_question(q, kg, Retriever(kg))
    assert "clasificación territorial" in result.text
    assert "Lengua viva A" not in result.text


def test_all_count_questions_call_groq(kg, monkeypatch):
    import rag_core
    calls = []

    def fake_groq(question, context, api_key, model):
        calls.append((question, context, model))
        return "El corpus registra 3 lenguas asociadas a Perú, no un censo actual. [G1]"

    monkeypatch.setattr(rag_core, "generate_groq", fake_groq)
    q = "¿Cuántas lenguas existen actualmente en Perú?"
    result = answer_question(q, kg, Retriever(kg), api_key="clave-ficticia")
    assert len(calls) == 1
    assert "HECHOS ESTRUCTURADOS" in calls[0][1]
    assert "3 registros" in calls[0][1]
    assert result.text.startswith("El corpus registra")
    assert "Groq" in result.method
    assert "alive1" in result.evidence


def test_all_list_questions_call_groq(kg, monkeypatch):
    import rag_core
    seen = []

    def fake_groq(question, context, api_key, model):
        seen.append(context)
        return "La lengua viva A aparece con una cifra de hablantes en el corpus [G1]."

    monkeypatch.setattr(rag_core, "generate_groq", fake_groq)
    result = answer_question("¿Qué lenguas se hablan actualmente en Perú?", kg,
                             Retriever(kg), api_key="clave-ficticia")
    assert seen and "HECHOS ESTRUCTURADOS" in seen[0]
    assert "Lengua viva A" in seen[0]
    assert "Groq" in result.method


def test_explanatory_question_calls_groq(kg, monkeypatch):
    import rag_core
    seen = []

    def fake_groq(question, context, api_key, model):
        seen.append((question, context))
        return "La familia quechua agrupa registros como Lengua viva A. [G1]"

    monkeypatch.setattr(rag_core, "generate_groq", fake_groq)
    result = answer_question("¿Qué sabemos de la familia Quechua?", kg,
                             Retriever(kg), api_key="clave-ficticia")
    assert len(seen) == 1 and "FICHAS RECUPERADAS" in seen[0][1]
    assert "Groq" in result.method


def test_groq_error_uses_clear_fallback(kg, monkeypatch):
    import rag_core

    def fail(*_args, **_kwargs):
        raise RuntimeError("proveedor desconectado")

    monkeypatch.setattr(rag_core, "generate_groq", fail)
    result = answer_question("¿Cuántas lenguas hay en Perú?", kg, Retriever(kg),
                             api_key="clave-ficticia")
    assert result.method == "Respaldo RDF (sin LLM)"
    assert "Groq no pudo" in result.notice
    assert "3 registros" in result.text


def test_unrecognized_citation_is_not_served(kg, monkeypatch):
    import rag_core
    monkeypatch.setattr(rag_core, "generate_groq", lambda *args: "Invento [G999].")
    result = answer_question("¿Cuántas lenguas hay en Perú?", kg,
                             Retriever(kg), api_key="clave-ficticia")
    assert "G999" not in result.text
    assert "Respaldo" in result.method
