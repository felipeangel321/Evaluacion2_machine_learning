"""La app demo debe transformar las canciones igual que el pipeline.

Si la transformación de app/core.py difiriera de data_transformation, las
métricas sobre el conjunto de test no coincidirían con las del informe.
Requiere haber ejecutado `kedro run` (modelos y datos procesados).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

pytestmark = pytest.mark.skipif(
    not (ROOT / "models/trained_models/gradient_boosting_clf.joblib").exists(),
    reason="Requiere los modelos generados por `kedro run`",
)


@pytest.fixture(scope="module")
def art():
    import core
    return core.load_artifacts()


def test_transformacion_igual_al_pipeline(art):
    import core
    X_app = core.to_model_input(art["test"], art)
    X_pipeline = pd.read_csv(ROOT / "data/05_model_input/X_test.csv")
    np.testing.assert_allclose(X_app.to_numpy(), X_pipeline.to_numpy(), atol=1e-9)


def test_metricas_reproducen_el_informe(art):
    import core
    pred = core.predict(art["test"], art)
    y_hit = (art["test"]["popularity"] >= core.HIT_THRESHOLD).astype(int)
    assert f1_score(y_hit, pred["prob_hit"] >= 0.5) == pytest.approx(0.680, abs=0.001)
    assert mean_absolute_error(art["test"]["popularity"], pred["pop_estimada"]) == pytest.approx(8.65, abs=0.01)
    curve = core.promotion_curve(pred, y_hit, budgets=[10])
    assert curve["% hits en lo promocionado"].iloc[0] == pytest.approx(85, abs=0.5)
