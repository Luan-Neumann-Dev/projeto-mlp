from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_ui_trains_evaluates_and_predicts() -> None:
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert len(app.tabs) == 4

    epochs = next(field for field in app.number_input if field.label == "Épocas")
    epochs.set_value(5).run()
    next(button for button in app.button if button.label == "Treinar rede").click().run()
    assert not app.exception
    assert any(metric.label == "MSE final de validação" for metric in app.metric)

    next(button for button in app.button if button.label == "Calcular MSE de teste").click().run()
    assert any(metric.label == "MSE de teste" for metric in app.metric)

    next(button for button in app.button if button.label == "Prever").click().run()
    assert not app.exception
    assert any(metric.label == "produtividade_kg_ha" for metric in app.metric)
