"""La v7.1.1 no debe fallar con traducciones previas de la v6.9 y v7.0."""
from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_compat_t():
    # La aplicación Streamlit no puede importarse en una prueba unitaria,
    # porque ejecuta st.set_page_config y los widgets en el nivel superior.
    module = ast.parse((ROOT / "app_lenguas.py").read_text(encoding="utf-8"))
    selected = [node for node in module.body
                if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id in ("LEGACY_TRANSLATIONS", "V71_COMPAT_TRANSLATIONS") for t in node.targets))
                or (isinstance(node, ast.FunctionDef) and node.name == "t")]
    assert len(selected) == 3
    scope = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "app_locale_part", "exec"),scope)
    return scope


def load_texts(raw: str):
    scope = {}
    exec(compile(raw, "historic_locale", "exec"), scope)
    return scope


@pytest.fixture(scope="module")
def handler():
    return load_compat_t()


@pytest.mark.parametrize("language", ["es", "pt", "en"])
def test_v71_missing_translation_with_v70_locale(handler, language):
    current = load_texts((ROOT / "locales.py").read_text(encoding="utf-8"))
    old = {k:v for k,v in current["STRINGS"][language].items()
           if k not in handler["V71_COMPAT_TRANSLATIONS"][language]}
    handler["_locale_t"] = lambda lang,key,**kwargs: old[key].format(**kwargs)
    expected = current["STRINGS"][language]
    for key in handler["V71_COMPAT_TRANSLATIONS"][language]:
        assert handler["t"](language,key, n=3, name="Kariri") == expected[key].format(n=3,name="Kariri")


@pytest.mark.parametrize("language", ["es", "pt", "en"])
def test_new_translation_uses_current_locale_when_available(handler,language):
    current = load_texts((ROOT / "locales.py").read_text(encoding="utf-8"))
    handler["_locale_t"] = current["t"]
    assert handler["t"](language,"explore_search_placeholder") == current["t"](language,"explore_search_placeholder")
    assert handler["t"](language,"graph_title",name="Kariri") == current["t"](language,"graph_title",name="Kariri")


def test_all_keys_for_three_languages_are_present():
    current = load_texts((ROOT / "locales.py").read_text(encoding="utf-8"))
    es_keys = set(current["STRINGS"]["es"])
    assert all(set(current["STRINGS"][language]) == es_keys for language in ("pt","en"))
    assert "explore_search_placeholder" in es_keys


def test_all_literal_t_keys_in_app_are_translated():
    current = load_texts((ROOT / "locales.py").read_text(encoding="utf-8"))
    source = ast.parse((ROOT / "app_lenguas.py").read_text(encoding="utf-8"))
    keys=[]
    for node in ast.walk(source):
        if not isinstance(node,ast.Call) or not node.args or not isinstance(node.func,ast.Name):
            continue
        if node.func.id == "t" and len(node.args)>1 and isinstance(node.args[1],ast.Constant) and isinstance(node.args[1].value,str):
            keys.append(node.args[1].value)
    for lang in ("es","pt","en"):
        assert set(keys).issubset(set(current["STRINGS"][lang])),(lang,set(keys)-set(current["STRINGS"][lang]))
