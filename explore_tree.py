"""Navegación encadenada sobre asociaciones existentes en el grafo RDF.

Una lengua puede estar asociada a varias agrupaciones. El grafo no dispone
necesariamente de aristas Family→Family: la inclusión de conjuntos de miembros
se usa SOLO para ordenar opciones de exploración, no para declarar una nueva
relación taxonómica.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rag_core import normalize


@dataclass
class ExplorerIndex:
    languages: dict
    families: dict
    members: dict[str, frozenset[str]]
    countries: dict[str, frozenset[str]]

    @classmethod
    def from_graph(cls, graph):
        languages = {e.id: e for e in graph.languages()}
        families = {e.id: e for e in graph.families()}
        members = {
            fid: frozenset(eid for eid in family.members if eid in languages)
            for fid, family in families.items()
        }
        countries = defaultdict(set)
        for e in languages.values():
            for country in e.countries:
                countries[country].add(e.id)
        return cls(languages, families, members,
                   {key: frozenset(value) for key, value in countries.items()})

    def country_ids(self, country: str) -> frozenset[str]:
        if country:
            return self.countries.get(country, frozenset())
        return frozenset(self.languages)

    def _unique_groups(self, ids: set[str] | frozenset[str]) -> list[str]:
        """Elimina asociaciones con conjunto idéntico y nombres redundantes.

        Sin descartar registros RDF: solo simplifica los menús de navegación.
        """
        group_by_members = defaultdict(list)
        for fid in ids:
            members = self.members[fid]
            if members:
                group_by_members[members].append(fid)
        return [min(similar, key=lambda fid:
                    (len(self.families[fid].label), normalize(self.families[fid].label), fid))
                for similar in group_by_members.values()]

    def main_groups(self, country: str) -> list[str]:
        """Agrupaciones más amplias con miembros en el país seleccionado."""
        eligible_ids = self.country_ids(country)
        active = self._unique_groups({fid for fid, members in self.members.items()
                                      if members & eligible_ids})
        roots = [fid for fid in active
                 if not any(self.members[fid] < self.members[other] for other in active)]
        return sorted(roots, key=lambda fid:
                      (-len(self.members[fid] & eligible_ids), normalize(self.families[fid].label)))

    def child_groups(self, parent: str, country: str) -> list[str]:
        """Subgrupos de un solo nivel, según inclusión de lenguas asociadas."""
        if parent not in self.members:
            return []
        eligible_ids = self.country_ids(country)
        parent_members = self.members[parent]
        children = self._unique_groups({fid for fid, members in self.members.items()
                                        if fid != parent and members < parent_members and
                                        bool(members & eligible_ids)})
        immediate = [fid for fid in children
                     if not any(self.members[fid] < self.members[other]
                                for other in children if other != fid)]
        return sorted(immediate, key=lambda fid:
                      (-len(self.members[fid] & eligible_ids), normalize(self.families[fid].label)))

    def language_ids(self, country: str, selected_group: str = "",
                     query: str = "") -> list[str]:
        ids = self.country_ids(country)
        if selected_group:
            ids = ids & self.members.get(selected_group, frozenset())
        if query.strip():
            term = normalize(query)
            ids = [eid for eid in ids
                   if term in normalize(self.languages[eid].label + " " + self.languages[eid].iso)]
        return sorted(ids, key=lambda eid: normalize(self.languages[eid].label))

    def ungrouped_ids(self, country: str) -> list[str]:
        """Lenguas del país sin ninguna asociación de familia registrada."""
        scope = self.country_ids(country)
        grouped = set().union(*self.members.values()) if self.members else set()
        return sorted(scope - grouped, key=lambda eid: normalize(self.languages[eid].label))

    def label(self, fid: str, country: str, *, count: bool = True) -> str:
        name = self.families[fid].label
        if not count:
            return name
        n = len(self.members[fid] & self.country_ids(country))
        return f"{name} ({n})"
