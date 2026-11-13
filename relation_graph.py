"""Vista local de relaciones RDF para RAG South America v7.0.

Sólo dibuja aristas existentes: lengua->familia, lengua->país y
familia->otras lenguas que declaran esa misma pertenencia. No usa
semanticallySimilarTo ni infiere nuevas jerarquías taxonómicas.

El modelo del grafo es puro y comprobable sin Streamlit ni Plotly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_core import KnowledgeGraph, normalize


@dataclass(frozen=True)
class GraphNode:
    key: str
    kind: str  # language | family | country
    value: str  # identificador RDF o texto del país
    label: str


@dataclass(frozen=True)
class GraphEdge:
    start: str
    end: str
    kind: str  # belongs_to | country | shares_family


@dataclass(frozen=True)
class LocalGraph:
    center: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    omitted: int = 0


def node_key(kind: str, value: str) -> str:
    # El prefijo evita confundir identificadores RDF con valores de países.
    return f"{kind}:{value}"


def build_local_graph(graph: KnowledgeGraph, kind: str, value: str,
                      max_families: int = 3, max_countries: int = 2,
                      max_related: int = 6, max_members: int = 9) -> LocalGraph:
    """Extrae una vecindad pequeña y determinista de relaciones verificables.

    Nunca modifica graph ni añade relaciones de similitud semántica.
    Un nodo de familia se conecta a las lenguas que declaran `belongsToFamily`.
    """
    if kind not in ("language", "family"):
        raise ValueError("Sólo se representan fichas de lengua o familia.")
    root = graph.entities.get(value)
    if root is None or root.kind.lower() != kind:
        raise ValueError("La ficha seleccionada no existe en el grafo.")

    nodes: dict[str, GraphNode] = {}
    edges: dict[tuple[str, str, str], GraphEdge] = {}

    def add(kind_: str, value_: str, label: str) -> str:
        key = node_key(kind_, value_)
        nodes.setdefault(key, GraphNode(key, kind_, value_, label))
        return key

    def connect(a: str, b: str, relationship: str) -> None:
        if a in nodes and b in nodes and a != b:
            edges[(a, b, relationship)] = GraphEdge(a, b, relationship)

    center = add(kind, root.id, root.label)
    omitted = 0
    if kind == "language":
        # Las familias pueden representar distintos niveles del mismo árbol:
        # representamos solamente las que están realmente relacionadas con la lengua.
        candidates = [graph.entities[fid] for fid in root.families
                      if fid in graph.entities and graph.entities[fid].kind == "Family"]
        candidates.sort(key=lambda f: (-len(f.members), normalize(f.label), f.id))
        selected_families = candidates[:max_families]
        omitted += max(len(candidates)-len(selected_families), 0)
        for fam in selected_families:
            fk = add("family", fam.id, fam.label)
            connect(center, fk, "belongs_to")
        countries = sorted(set(root.countries), key=normalize)
        omitted += max(len(countries)-max_countries, 0)
        for country in countries[:max_countries]:
            ck = add("country", country, country)
            connect(center, ck, "country")

        # Una lengua relacionada se muestra conectada a la familia COMPARTIDA;
        # nunca se traza una arista lengua->lengua de parentesco no registrada.
        # Distribución por rondas para no mostrar sólo una rama de un gran grupo.
        pools: dict[str, list[str]] = {}
        for fam in selected_families:
            mids = (mid for mid in fam.members if mid != root.id
                    and mid in graph.entities and graph.entities[mid].kind == "Language")
            pools[fam.id] = sorted(set(mids), key=lambda mid: (normalize(graph.entities[mid].label), mid))
        shown: set[str] = set()
        while len(shown) < max_related and any(pools.values()):
            progressed = False
            for fam in selected_families:
                pool = pools[fam.id]
                while pool and pool[0] in shown:
                    pool.pop(0)
                if not pool:
                    continue
                mid = pool.pop(0)
                if mid not in shown:
                    lk = add("language", mid, graph.entities[mid].label)
                    shown.add(mid)
                    progressed = True
                else:
                    lk = node_key("language", mid)
                connect(node_key("family", fam.id), lk, "shares_family")
                if len(shown) >= max_related:
                    break
            if not progressed:
                break
        # Si dos grupos seleccionados contienen la misma lengua representada,
        # sus dos enlaces RDF pueden mostrarse en el mismo grafo local.
        for fam in selected_families:
            fk = node_key("family", fam.id)
            for mid in shown:
                if mid in fam.members:
                    connect(fk, node_key("language", mid), "shares_family")
        other_members = set().union(*(set(f.members) for f in selected_families)) if selected_families else set()
        omitted += max(len(other_members - {root.id} - shown), 0)
    else:
        members = [graph.entities[mid] for mid in root.members
                   if mid in graph.entities and graph.entities[mid].kind == "Language"]
        members.sort(key=lambda e: (normalize(e.label), e.id))
        selected = members[:max_members]
        omitted = max(len(members)-len(selected), 0)
        for member in selected:
            lk = add("language", member.id, member.label)
            connect(center, lk, "shares_family")

    return LocalGraph(center, tuple(nodes.values()), tuple(edges.values()), omitted)


def graph_targets(view: LocalGraph) -> dict[str, tuple[str, str]]:
    return {node.key: (node.kind, node.value) for node in view.nodes}


def selected_node(event: Any, view: LocalGraph) -> GraphNode | None:
    """Extrae la ficha clicada desde `st.plotly_chart(..., on_select='rerun')`."""
    try:
        selection = event.get("selection", {})
        points = selection.get("points", [])
        if not points:
            return None
        # Nos interesa la selección más reciente; sólo aceptamos nodos del grafo.
        current = {node.key: node for node in view.nodes}
        for point in reversed(points):
            data = point.get("customdata", [])
            if data and str(data[0]) in current:
                return current[str(data[0])]
    except (AttributeError, ValueError, IndexError, TypeError, KeyError):
        pass
    return None


def _short_label(value: str, max_length: int = 18) -> str:
    value = " ".join(value.split())
    return value if len(value) <= max_length else value[:max_length-1].rstrip() + "…"


def node_coordinates(view: LocalGraph) -> dict[str, tuple[float, float]]:
    """Distribución fija para pocos nodos: estable en cada rerun y responsiva."""
    center_node = next(node for node in view.nodes if node.key == view.center)
    other_nodes = [n for n in view.nodes if n.key != view.center]
    positions = {view.center: (0.0, 0.0)}
    if center_node.kind == "family":
        n = len(other_nodes)
        for i, node in enumerate(other_nodes):
            # Alternar dos columnas en grupos grandes reduce solapamientos.
            col = i % 2
            row = i // 2
            positions[node.key] = (-1.35 if col == 0 else 1.35,
                                   1.3 - row * 0.72)
        return positions

    family = [n for n in other_nodes if n.kind == "family"]
    country = [n for n in other_nodes if n.kind == "country"]
    langs = [n for n in other_nodes if n.kind == "language"]
    for i, node in enumerate(family):
        positions[node.key] = ((i - (len(family)-1)/2)*1.0, 1.55)
    for i, node in enumerate(country):
        positions[node.key] = (-1.65, -0.35 - i*0.8)
    for i, node in enumerate(langs):
        positions[node.key] = (1.75, 1.0 - i*0.65)
    return positions


def build_figure(view: LocalGraph, language: str = "es"):
    """La dependencia visual se carga sólo al mostrar una red."""
    import plotly.graph_objects as go

    pos = node_coordinates(view)
    fig = go.Figure()
    edge_color = "#46545e"
    for edge in view.edges:
        x1, y1 = pos[edge.start]
        x2, y2 = pos[edge.end]
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2], mode="lines", hoverinfo="skip",
            line=dict(width=1.5, color=edge_color), showlegend=False,
            name=""))

    labels = {
        "es": {"language": "Lengua", "family": "Familia o subgrupo", "country": "País"},
        "pt": {"language": "Língua", "family": "Família ou subgrupo", "country": "País"},
        "en": {"language": "Language", "family": "Family or subgroup", "country": "Country"},
    }.get(language, {})
    colors = {"language": "#477bad", "family": "#a98b50", "country": "#798898"}
    for kind in ("language", "family", "country"):
        group = [node for node in view.nodes if node.kind == kind and node.key != view.center]
        if not group:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[node.key][0] for node in group],
            y=[pos[node.key][1] for node in group],
            mode="markers+text", text=[_short_label(node.label) for node in group],
            textposition="bottom center", textfont=dict(size=11, color="#d5dde5"),
            customdata=[[node.key] for node in group],
            hovertext=[node.label for node in group], hovertemplate="<b>%{hovertext}</b><extra></extra>",
            marker=dict(size=19, color=colors[kind], line=dict(color="#101820", width=1)),
            showlegend=False, name=labels.get(kind, kind),
        ))
    root = next(node for node in view.nodes if node.key == view.center)
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text", text=[_short_label(root.label, 22)],
        textposition="top center", textfont=dict(size=13, color="#ecf5fa"),
        hovertext=[root.label], hovertemplate="<b>%{hovertext}</b><extra></extra>",
        customdata=[[root.key]],
        marker=dict(size=31, color="#2ea043", line=dict(color="#c6f1d0", width=2)),
        showlegend=False, name=root.label,
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#cbd5df"),
        height=430, margin=dict(l=16, r=16, t=24, b=16),
        showlegend=False, clickmode="event+select", dragmode="pan",
        hovermode="closest", uirevision=view.center,
        xaxis=dict(visible=False, range=[-2.65, 2.75], fixedrange=False),
        yaxis=dict(visible=False, range=[-2.55, 2.45], fixedrange=False,
                   scaleanchor="x", scaleratio=1),
    )
    return fig


def display_local_graph(graph: KnowledgeGraph, kind: str, value: str,
                        language: str, t, country_label, open_in_explorer,
                        scope: str = "explore", expanded: bool = False):
    """Control Streamlit opcional: clic de Plotly y selector accesible móvil."""
    import streamlit as st

    headings = {
        "es": {"language": "Lengua", "family": "Familia o subgrupo", "country": "País"},
        "pt": {"language": "Língua", "family": "Família ou subgrupo", "country": "País"},
        "en": {"language": "Language", "family": "Family or subgroup", "country": "Country"},
    }
    scope_key = f"graph_anchor_{scope}"
    center_key = f"graph_center_{scope}"
    seed = node_key(kind, value)
    if st.session_state.get(scope_key) != seed:
        st.session_state[scope_key] = seed
        st.session_state[center_key] = seed
    desired_center = st.session_state[center_key]
    center_kind, center_value = desired_center.split(":", 1)
    if center_kind not in ("language", "family"):
        center_kind, center_value = kind, value
    view = build_local_graph(graph, center_kind, center_value)
    root = next(node for node in view.nodes if node.key == view.center)

    with st.expander(t(language, "graph_title", name=root.label), expanded=expanded):
        st.caption(t(language, "graph_help"))
        st.caption(t(language, "graph_legend"))
        fig = build_figure(view, language)
        state = st.plotly_chart(
            fig, key=f"v70_graph_{scope}_{view.center}", on_select="rerun",
            selection_mode="points", width="stretch", theme=None,
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": "reset"},
            alt=t(language, "graph_alt", name=root.label),
        )
        node = selected_node(state, view)
        options = ["" ] + [n.key for n in view.nodes]
        fallback = st.selectbox(
            t(language, "graph_node_select"), options,
            format_func=lambda key: (t(language, "graph_node_placeholder") if not key else
                                     f"{headings[language][next(n.kind for n in view.nodes if n.key == key)]}: "
                                     f"{next(n.label for n in view.nodes if n.key == key)}"),
            key=f"graph_selector_{scope}_{view.center}_{language}",
        )
        if fallback:
            node = next((n for n in view.nodes if n.key == fallback), None)
        if view.omitted:
            st.caption(t(language, "graph_limited", n=view.omitted))
        if len(view.nodes) == 1:
            st.info(t(language, "graph_no_relations"))
        if node:
            name = country_label(language, node.value) if node.kind == "country" else node.label
            st.caption(t(language, "graph_selected", name=name))
            # Botones apilados, cómodos para pantallas estrechas.
            st.button(t(language, "graph_open"),
                      on_click=open_in_explorer, args=(node.kind, node.value),
                      key=f"open_graph_node_{scope}_{view.center}_{node.key}",
                      use_container_width=True)
            if node.kind in ("language", "family") and node.key != view.center:
                st.button(t(language, "graph_center"),
                          on_click=lambda: st.session_state.update({center_key: node.key}),
                          key=f"center_graph_node_{scope}_{view.center}_{node.key}",
                          use_container_width=True)
        else:
            st.caption(t(language, "graph_click_help"))
