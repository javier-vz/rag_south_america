"""Regresión: app v6.6 + locales v6.4 no debe fallar por las 19 claves nuevas."""
from __future__ import annotations

import ast
from pathlib import Path
import string

from locales import STRINGS, t as live_t

ROOT = Path(__file__).resolve().parents[1]


def _bridge_with_legacy_locale():
    module = ast.parse((ROOT / 'app_lenguas.py').read_text(encoding='utf-8'))
    assignment = next(n for n in module.body if isinstance(n, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'LEGACY_TRANSLATIONS'
                              for target in n.targets))
    bridge = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 't')
    namespace = {'_locale_t': lambda lang, key, **kwargs: (_ for _ in ()).throw(KeyError(key))}
    tiny_module = ast.Module(body=[assignment, bridge], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tiny_module), '<translation_bridge>', 'exec'), namespace)
    return namespace['t'], namespace['LEGACY_TRANSLATIONS']


def _kwargs(template):
    placeholders = [name for _, name, _, _ in string.Formatter().parse(template) if name]
    return {name: 2 if name == 'n' else 'prueba' for name in placeholders}


def test_country_select_no_longer_crashes_with_old_locales():
    t, _ = _bridge_with_legacy_locale()
    assert t('es', 'explore_country_placeholder') == 'Selecciona un país…'
    assert t('pt', 'explore_country_placeholder') == 'Selecione um país…'


def test_all_new_explorer_and_about_copy_survives_mixed_versions():
    t, fallback = _bridge_with_legacy_locale()
    assert len(fallback['es']) == len(fallback['pt']) == 19
    assert set(fallback['es']) == set(fallback['pt'])
    for lang in ('es', 'pt'):
        for key, template in fallback[lang].items():
            output = t(lang, key, **_kwargs(template))
            assert output and output != key
            assert output == STRINGS[lang][key].format(**_kwargs(template))


def test_unknown_key_has_actionable_error():
    t, _ = _bridge_with_legacy_locale()
    try:
        t('pt', 'new_field_from_future_version')
    except KeyError as error:
        assert 'locales.py' in str(error)
    else:
        raise AssertionError('La clave desconocida no debería pasar inadvertida')


def test_current_locales_fallback_when_portuguese_key_missing(monkeypatch):
    monkeypatch.delitem(STRINGS['pt'], 'footer')
    assert live_t('pt', 'footer') == STRINGS['es']['footer']
