"""
Tests de los nodos de optimización, ensambles y generalización.
Usan datos sintéticos pequeños para correr en segundos. Se ejecutan con: pytest
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from evaluacion2_machine.pipelines.ensembles.nodes import (
    build_classification_ensembles,
    build_regression_ensembles,
    stacking_weights,
)
from evaluacion2_machine.pipelines.evaluation.nodes import _diagnosis
from evaluacion2_machine.pipelines.hyperparameter_tuning.nodes import _tune

PARAMS = {"random_state": 0, "ensembles": {"voting": "soft", "stacking_cv": 3}}


@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=300, n_features=8, weights=[0.7], random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.DataFrame({"is_hit": y})


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=300, n_features=8, noise=10, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.DataFrame({"popularity": y})


class TestTuning:
    def test_tuning_never_returns_worse_than_initial(self, clf_data):
        X, y = clf_data
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        initial = LogisticRegression(C=1.0, max_iter=1000)
        # Espacio de búsqueda que solo contiene un C pésimo: la búsqueda pierde
        search = GridSearchCV(LogisticRegression(max_iter=1000), {"C": [1e-6]}, scoring="f1", cv=cv)
        model, row = _tune("LR", initial, search, X, y.squeeze(), cv, "f1", lower_is_better=False)

        assert row["Modelo_final"].startswith("inicial")
        assert model.get_params()["C"] == 1.0
        assert row["Optimizado"] < row["Inicial"]

    def test_tuning_reports_mae_as_positive(self, reg_data):
        X, y = reg_data
        cv = KFold(n_splits=3, shuffle=True, random_state=0)
        search = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0]}, scoring="neg_mean_absolute_error", cv=cv)
        _, row = _tune("Ridge", Ridge(alpha=1.0), search, X, y.squeeze(), cv,
                       "neg_mean_absolute_error", lower_is_better=True)

        assert row["Metrica_CV"] == "MAE"
        assert row["Inicial"] > 0 and row["Optimizado"] > 0
        assert row["Mejora"] >= 0


def _clf_base():
    return (LogisticRegression(max_iter=1000), RandomForestClassifier(n_estimators=10, random_state=0),
            HistGradientBoostingClassifier(max_iter=20, random_state=0))


def _reg_base():
    return (Ridge(), RandomForestRegressor(n_estimators=10, random_state=0),
            HistGradientBoostingRegressor(max_iter=20, random_state=0))


class TestEnsembles:
    def test_classification_ensembles(self, clf_data):
        X, y = clf_data
        voting, stacking = build_classification_ensembles(X, y, PARAMS, *_clf_base())

        for model in (voting, stacking):
            proba = model.predict_proba(X)[:, 1]
            assert ((proba >= 0) & (proba <= 1)).all()

    def test_regression_ensembles(self, reg_data):
        X, y = reg_data
        voting, stacking = build_regression_ensembles(X, y, PARAMS, *_reg_base())

        assert len(voting.predict(X)) == len(stacking.predict(X)) == len(X)

    def test_stacking_weights(self, clf_data, reg_data):
        _, stack_clf = build_classification_ensembles(*clf_data, PARAMS, *_clf_base())
        _, stack_reg = build_regression_ensembles(*reg_data, PARAMS, *_reg_base())
        weights = stacking_weights(stack_clf, stack_reg)

        assert len(weights) == 6
        assert list(weights["Modelo_base"][:3]) == ["lineal", "random_forest", "gradient_boosting"]
        assert np.isfinite(weights["Peso_meta_modelo"]).all()


@pytest.mark.parametrize("gap, expected", [
    (0.02, "generaliza bien"), (0.15, "sobreajuste leve"), (0.40, "sobreajuste"),
])
def test_generalization_diagnosis(gap, expected):
    assert _diagnosis(gap) == expected
