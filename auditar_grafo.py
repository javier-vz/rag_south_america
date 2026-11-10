#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auditoría reproducible del RDF, SIN modificar el grafo original.

Uso:
    python auditar_grafo.py

Produce: auditoria_datos.json y auditoria_lenguas.csv
Las alertas de clasificación son pistas de revisión, no decisiones científicas.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from rdflib import Graph

from rag_core import EX, KnowledgeGraph, normalize

ROOT = Path(__file__).resolve().parent
TTL = ROOT / "grafo_enriquecido.ttl"


def audit(path: Path, dest: Path):
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path.name}. Ejecuta primero python descargar_datos.py")
    rdf = Graph()
    rdf.parse(str(path), format="turtle")
    graph = KnowledgeGraph.from_turtle(path)
    langs = graph.languages()
    families = graph.families()
    # No agregar distintos países para inferir ausencia transfronteriza.
    by_country = Counter(country for e in langs for country in e.countries)
    raw_links = list(rdf.objects(None, EX.describedByResource))
    wrapped_links = sum(str(uri).startswith(str(EX) + "https://") or
                        str(uri).startswith(str(EX) + "http://") for uri in raw_links)
    with_docs = sum(bool(e.wiki or e.description) for e in langs)
    summary = {
        "graph_triples": len(rdf),
        "language_records": len(langs),
        "family_or_subgroup_records": len(families),
        "without_speaker_figure": sum(e.speakers is None for e in langs),
        "speaker_figure_zero": sum(e.speakers == 0 for e in langs),
        "speaker_figure_positive": sum(e.speakers is not None and e.speakers > 0 for e in langs),
        "without_country": sum(not e.countries for e in langs),
        "without_family_or_subgroup": sum(not e.families for e in langs),
        "with_iso": sum(bool(e.iso) for e in langs),
        "with_enriched_comment": sum(bool(e.comment) for e in langs),
        "with_wikipedia_or_wikidata_description": with_docs,
        "country_values": dict(by_country.most_common()),
        "peru_records": len(graph.country("Perú")),
        "peru_without_speakers": sum(e.speakers is None for e in graph.country("Perú")),
        "source_links_total": len(raw_links),
        "source_links_embedded_in_local_uris": wrapped_links,
        "obvious_non_indigenous_labels_for_review": [
            {"id": e.id, "label": e.label} for e in langs if e.needs_indigenous_review
        ],
        "limitations": [
            "No se infiere actualidad o extinción desde una cifra ausente o cero.",
            "Los valores de hablantes no presentan una fecha comparable ni una metodología homogénea.",
            "ex:Family incluye agrupaciones y niveles taxonómicos; no equivale a familias independientes.",
            "La clase ex:Language no garantiza por sí sola que cada ficha sea indígena.",
            "ex:wikidata_countries no permite medir cobertura transfronteriza.",
            "Los rdfs:comment enriquecidos con IA no son verificación independiente.",
            "Los enlaces encapsulados bajo http://example.org/lang/ necesitan normalización para ser navegables.",
        ],
    }
    dest.mkdir(parents=True, exist_ok=True)
    json_out = dest / "auditoria_datos.json"
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_out = dest / "auditoria_lenguas.csv"
    with csv_out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "nombre", "iso", "paises", "cifra_hablantes_sin_fecha",
                         "familias_y_subgrupos", "tiene_resumen_generado_IA", "fuentes_n",
                         "alerta_clasificacion_manual"])
        for e in sorted(langs, key=lambda e: normalize(e.label)):
            writer.writerow([e.id, e.label, e.iso, "; ".join(e.countries),
                             "" if e.speakers is None else e.speakers, "; ".join(e.families),
                             bool(e.comment), len(e.sources), bool(e.needs_indigenous_review)])
    return summary, csv_out, json_out


if __name__ == "__main__":
    data, csv_path, json_path = audit(TTL, ROOT)
    print(json.dumps({k: v for k, v in data.items() if k not in
                      ("country_values", "obvious_non_indigenous_labels_for_review", "limitations")},
                     ensure_ascii=False, indent=2))
    print("\nArchivos creados (sin alterar el TTL):", json_path.name, csv_path.name)
