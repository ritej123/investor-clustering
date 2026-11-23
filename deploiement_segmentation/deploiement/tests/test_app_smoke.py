"""Test de fumée de l'interface (nécessite streamlit installé)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Synthèse", "Profils des clusters", "Cartographie", "Simulateur",
         "Comparaison des modèles", "Données et export", "Méthodologie"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_error(page):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
