import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score, f1_score, mean_absolute_error, precision_score,
    r2_score, recall_score, roc_auc_score, roc_curve, root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, learning_curve

logger = logging.getLogger(__name__)

CLF_NAMES = ["Baseline", "LogisticRegression", "RandomForest", "GradientBoosting", "Voting", "Stacking"]
REG_NAMES = ["Baseline", "Ridge", "RandomForest", "GradientBoosting", "Voting", "Stacking"]
COLORS = {"Baseline": "#95A5A6", "LogisticRegression": "#3498DB", "Ridge": "#3498DB",
          "RandomForest": "#E67E22", "GradientBoosting": "#1DB954",
          "Voting": "#9B59B6", "Stacking": "#2C3E50"}


# ─────────────────────────────────────────────
# Clasificación
# ─────────────────────────────────────────────

def evaluate_classifiers(X_test, y_test_clf, baseline, log_reg, rf, gb, voting, stacking) -> tuple:
    # F1 es la métrica principal: combina Precision (de las canciones que el
    # modelo marca como hit, cuántas lo son) y Recall (de los hits reales,
    # cuántos detecta). ROC-AUC mide la capacidad de ordenar canciones por
    # probabilidad de hit, independiente del umbral de decisión.
    y = y_test_clf.squeeze()
    models = dict(zip(CLF_NAMES, [baseline, log_reg, rf, gb, voting, stacking]))

    rows = []
    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred),
            "ROC_AUC": roc_auc_score(y, proba),
        })
    metrics = pd.DataFrame(rows).round(4)
    logger.info("Métricas clasificación (test):\n%s", metrics.to_string(index=False))

    fig_cm, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (name, model) in zip(axes.ravel(), models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y, ax=ax, cmap="Greens", colorbar=False,
            display_labels=["No hit", "Hit"], values_format=",",
        )
        ax.set_title(name)
        ax.grid(False)
    fig_cm.suptitle("Matrices de confusión — test")
    fig_cm.tight_layout()
    plt.close(fig_cm)

    fig_roc, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y, model.predict_proba(X_test)[:, 1])
        auc = metrics.loc[metrics["Modelo"] == name, "ROC_AUC"].iloc[0]
        ax.plot(fpr, tpr, color=COLORS[name], label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos (Recall)")
    ax.legend(loc="lower right")
    ax.set_title("Curvas ROC — test")
    fig_roc.tight_layout()
    plt.close(fig_roc)

    return metrics, fig_cm, fig_roc


def cross_validate_classifiers(X_train, y_train_clf, params, log_reg, rf, gb, voting, stacking) -> pd.DataFrame:
    # Validación cruzada estratificada sobre train: confirma que el desempeño
    # en test no depende de un split afortunado. Una std baja entre folds
    # indica un modelo estable. Se usa clone() para reentrenar desde cero con
    # los mismos hiperparámetros en cada fold.
    y = y_train_clf.squeeze()
    cv = StratifiedKFold(n_splits=params["cv_folds"], shuffle=True, random_state=params["random_state"])
    scoring = ["f1", "recall", "precision", "roc_auc"]

    rows = []
    for name, model in zip(CLF_NAMES[1:], [log_reg, rf, gb, voting, stacking]):
        scores = cross_validate(clone(model), X_train, y, cv=cv, scoring=scoring)
        row = {"Modelo": name}
        for s in scoring:
            row[f"{s}_mean"] = scores[f"test_{s}"].mean()
            row[f"{s}_std"] = scores[f"test_{s}"].std()
        rows.append(row)
        logger.info("CV %s — F1: %.4f ± %.4f", name, row["f1_mean"], row["f1_std"])
    return pd.DataFrame(rows).round(4)


# ─────────────────────────────────────────────
# Regresión
# ─────────────────────────────────────────────

def evaluate_regressors(X_test, y_test_reg, baseline, ridge, rf, gb, voting, stacking) -> tuple:
    # MAE: error promedio en puntos de popularidad (métrica principal, la más
    # fácil de explicar al negocio). RMSE penaliza más los errores grandes.
    # R²: proporción de la variabilidad de popularity que explica el modelo.
    y = y_test_reg.squeeze()
    models = dict(zip(REG_NAMES, [baseline, ridge, rf, gb, voting, stacking]))

    rows, preds = [], {}
    for name, model in models.items():
        pred = model.predict(X_test)
        preds[name] = pred
        rows.append({
            "Modelo": name,
            "MAE": mean_absolute_error(y, pred),
            "RMSE": root_mean_squared_error(y, pred),
            "R2": r2_score(y, pred),
        })
    metrics = pd.DataFrame(rows).round(4)
    logger.info("Métricas regresión (test):\n%s", metrics.to_string(index=False))

    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(3000, len(y)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    axes[-1].axis("off")
    for ax, name in zip(axes, REG_NAMES[1:]):
        ax.scatter(y.iloc[idx], preds[name][idx], s=6, alpha=0.3, color=COLORS[name])
        ax.plot([0, 100], [0, 100], "r--", lw=1)
        r2 = metrics.loc[metrics["Modelo"] == name, "R2"].iloc[0]
        ax.set_title(f"{name} — R² = {r2:.3f}")
        ax.set_xlabel("popularity real")
        ax.set_ylabel("popularity predicha")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
    fig.suptitle("Real vs predicho — test (muestra de 3.000 canciones)")
    fig.tight_layout()
    plt.close(fig)

    return metrics, fig


def cross_validate_regressors(X_train, y_train_reg, params, ridge, rf, gb, voting, stacking) -> pd.DataFrame:
    y = y_train_reg.squeeze()
    cv = KFold(n_splits=params["cv_folds"], shuffle=True, random_state=params["random_state"])
    scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}

    rows = []
    for name, model in zip(REG_NAMES[1:], [ridge, rf, gb, voting, stacking]):
        scores = cross_validate(clone(model), X_train, y, cv=cv, scoring=scoring)
        row = {"Modelo": name}
        for s in scoring:
            vals = scores[f"test_{s}"]
            vals = -vals if s != "R2" else vals
            row[f"{s}_mean"] = vals.mean()
            row[f"{s}_std"] = vals.std()
        rows.append(row)
        logger.info("CV %s — MAE: %.3f ± %.3f", name, row["MAE_mean"], row["MAE_std"])
    return pd.DataFrame(rows).round(4)


def plot_metrics_comparison(clf_metrics: pd.DataFrame, reg_metrics: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    clf_metrics.set_index("Modelo")[["Precision", "Recall", "F1", "ROC_AUC"]].plot.bar(ax=axes[0], rot=15)
    axes[0].set_title("Clasificación (is_hit) — métricas en test")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper left")

    reg_metrics.set_index("Modelo")[["MAE", "RMSE"]].plot.bar(ax=axes[1], rot=15, color=["#1DB954", "#E67E22"])
    axes[1].set_title("Regresión (popularity) — error en test (menor es mejor)")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────
# Interpretación
# ─────────────────────────────────────────────

def _feature_group(col: str) -> str:
    for prefix in ("track_genre_", "key_", "time_signature_"):
        if col.startswith(prefix):
            return prefix[:-1]
    return col


def grouped_permutation_importance(X_test, y_test_clf, y_test_reg, gb_clf, gb_reg, params) -> tuple:
    # Importancia por permutación: se desordena una variable y se mide cuánto
    # empeora el modelo. Mide el aporte real de cada variable al modelo final
    # (el mejor, GradientBoosting), no solo cuántas veces se usa en los árboles.
    # Las 114 columnas One-Hot de género se permutan JUNTAS: permutarlas por
    # separado subestimaría el efecto del género (cada una aporta poco sola).
    rng = np.random.default_rng(params["random_state"])
    n_repeats = params["permutation_repeats"]
    groups = pd.Series(X_test.columns, index=X_test.columns).map(_feature_group)
    y_clf, y_reg = y_test_clf.squeeze().values, y_test_reg.squeeze().values

    base_auc = roc_auc_score(y_clf, gb_clf.predict_proba(X_test)[:, 1])
    base_mae = mean_absolute_error(y_reg, gb_reg.predict(X_test))

    rows = []
    for group in groups.unique():
        cols = groups[groups == group].index
        auc_drop, mae_rise = [], []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[cols] = X_test[cols].values[rng.permutation(len(X_test))]
            auc_drop.append(base_auc - roc_auc_score(y_clf, gb_clf.predict_proba(X_perm)[:, 1]))
            mae_rise.append(mean_absolute_error(y_reg, gb_reg.predict(X_perm)) - base_mae)
        rows.append({"Variable": group,
                     "caida_ROC_AUC_clf": np.mean(auc_drop),
                     "aumento_MAE_reg": np.mean(mae_rise)})

    importance = pd.DataFrame(rows).sort_values("caida_ROC_AUC_clf", ascending=False).round(4)
    logger.info("Importancia por permutación:\n%s", importance.head(8).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top = importance.head(12)
    sns.barplot(data=top, y="Variable", x="caida_ROC_AUC_clf", ax=axes[0], color="#1DB954")
    axes[0].set_title("Clasificación: caída de ROC-AUC al permutar")
    top_reg = importance.sort_values("aumento_MAE_reg", ascending=False).head(12)
    sns.barplot(data=top_reg, y="Variable", x="aumento_MAE_reg", ax=axes[1], color="#E67E22")
    axes[1].set_title("Regresión: aumento de MAE al permutar")
    fig.suptitle("Importancia de variables — GradientBoosting (test)")
    fig.tight_layout()
    plt.close(fig)

    return importance, fig


def logistic_coefficients(log_reg, X_test) -> tuple:
    # Los coeficientes de la regresión logística (features escaladas) indican
    # la DIRECCIÓN del efecto: positivo = aumenta la probabilidad de hit.
    # Complementa la importancia por permutación, que solo indica magnitud.
    coefs = pd.DataFrame({"Variable": X_test.columns, "Coeficiente": log_reg.coef_[0]})
    coefs["Tipo"] = coefs["Variable"].map(_feature_group)

    audio = coefs[coefs["Tipo"] == coefs["Variable"]].sort_values("Coeficiente")
    genres = coefs[coefs["Tipo"] == "track_genre"].sort_values("Coeficiente")
    genres_top = pd.concat([genres.head(10), genres.tail(10)])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].barh(audio["Variable"], audio["Coeficiente"],
                 color=np.where(audio["Coeficiente"] > 0, "#1DB954", "#E74C3C"))
    axes[0].set_title("Features de audio — efecto sobre P(hit)")
    axes[1].barh(genres_top["Variable"].str.replace("track_genre_", ""), genres_top["Coeficiente"],
                 color=np.where(genres_top["Coeficiente"] > 0, "#1DB954", "#E74C3C"))
    axes[1].set_title("Géneros — 10 más y 10 menos propensos a hit")
    fig.suptitle("Coeficientes de LogisticRegression")
    fig.tight_layout()
    plt.close(fig)

    return coefs.sort_values("Coeficiente", ascending=False).round(4), fig


# ─────────────────────────────────────────────
# Capacidad de generalización
# ─────────────────────────────────────────────
#
# Un modelo generaliza bien si su desempeño con datos que NO vio (validación
# cruzada y test) es parecido al que obtiene con los datos de entrenamiento.
# Se comparan tres mediciones por modelo:
#   - Train: datos con los que se entrenó (optimista por definición).
#   - CV:    promedio de 5 folds sobre train, cada fold evaluado sin haberlo visto.
#   - Test:  20% reservado desde el inicio, nunca usado para entrenar ni ajustar.
# Brecha train → test grande = sobreajuste (memoriza en vez de aprender).
# CV ≈ test = la estimación es confiable y no depende de un split afortunado.

def _diagnosis(rel_gap: float) -> str:
    if rel_gap < 0.10:
        return "generaliza bien"
    if rel_gap < 0.25:
        return "sobreajuste leve"
    return "sobreajuste"


def generalization_classifiers(X_train, y_train_clf, clf_test_metrics, cv_clf_results,
                               log_reg, rf, gb, voting, stacking) -> pd.DataFrame:
    y = y_train_clf.squeeze()
    test = clf_test_metrics.set_index("Modelo")
    cv = cv_clf_results.set_index("Modelo")

    rows = []
    for name, model in zip(CLF_NAMES[1:], [log_reg, rf, gb, voting, stacking]):
        f1_train = f1_score(y, model.predict(X_train))
        auc_train = roc_auc_score(y, model.predict_proba(X_train)[:, 1])
        f1_test = test.loc[name, "F1"]
        rel_gap = (f1_train - f1_test) / f1_train
        rows.append({
            "Modelo": name,
            "F1_train": f1_train, "F1_cv": cv.loc[name, "f1_mean"], "F1_test": f1_test,
            "Brecha_F1": f1_train - f1_test,
            "ROC_AUC_train": auc_train, "ROC_AUC_cv": cv.loc[name, "roc_auc_mean"],
            "ROC_AUC_test": test.loc[name, "ROC_AUC"],
            "Diagnostico": _diagnosis(rel_gap),
        })
    result = pd.DataFrame(rows).round(4)
    logger.info("Generalización (clasificación):\n%s", result.to_string(index=False))
    return result


def generalization_regressors(X_train, y_train_reg, reg_test_metrics, cv_reg_results,
                              ridge, rf, gb, voting, stacking) -> pd.DataFrame:
    y = y_train_reg.squeeze()
    test = reg_test_metrics.set_index("Modelo")
    cv = cv_reg_results.set_index("Modelo")

    rows = []
    for name, model in zip(REG_NAMES[1:], [ridge, rf, gb, voting, stacking]):
        pred = model.predict(X_train)
        mae_train = mean_absolute_error(y, pred)
        mae_test = test.loc[name, "MAE"]
        rel_gap = (mae_test - mae_train) / mae_test
        rows.append({
            "Modelo": name,
            "MAE_train": mae_train, "MAE_cv": cv.loc[name, "MAE_mean"], "MAE_test": mae_test,
            "Brecha_MAE": mae_test - mae_train,
            "R2_train": r2_score(y, pred), "R2_cv": cv.loc[name, "R2_mean"], "R2_test": test.loc[name, "R2"],
            "Diagnostico": _diagnosis(rel_gap),
        })
    result = pd.DataFrame(rows).round(4)
    logger.info("Generalización (regresión):\n%s", result.to_string(index=False))
    return result


def plot_generalization(gen_clf: pd.DataFrame, gen_reg: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
    palette = ["#BDC3C7", "#3498DB", "#1DB954"]

    gen_clf.set_index("Modelo")[["F1_train", "F1_cv", "F1_test"]].plot.bar(ax=axes[0], rot=15, color=palette)
    axes[0].set_title("Clasificación — F1 en train, validación cruzada y test (mayor es mejor)")
    axes[0].set_ylim(0, 1)
    axes[0].legend(["Train", "CV (5 folds)", "Test"], loc="lower right")

    gen_reg.set_index("Modelo")[["MAE_train", "MAE_cv", "MAE_test"]].plot.bar(ax=axes[1], rot=15, color=palette)
    axes[1].set_title("Regresión — MAE en train, validación cruzada y test (menor es mejor)")
    axes[1].legend(["Train", "CV (5 folds)", "Test"], loc="lower right")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f", fontsize=8)

    fig.suptitle("Capacidad de generalización: una brecha pequeña entre train y test indica que el modelo no memoriza")
    fig.tight_layout()
    plt.close(fig)
    return fig


def learning_curves(X_train, y_train_clf, y_train_reg, gb_clf, gb_reg, params) -> tuple:
    # Curva de aprendizaje del modelo base del ensamble (Gradient Boosting):
    # entrena con porciones crecientes de train y mide el score en validación.
    # - Si train y validación convergen → el modelo no sobreajusta.
    # - Si la validación sigue subiendo al final → más datos ayudarían.
    # - Si ambas se estancan bajas → falta información (variables), no datos.
    lc, rs = params["learning_curve"], params["random_state"]
    sizes = np.array(lc["train_sizes"])

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    tasks = [
        ("clasificacion", gb_clf, y_train_clf, "f1", "F1",
         StratifiedKFold(n_splits=lc["cv_folds"], shuffle=True, random_state=rs), 1),
        ("regresion", gb_reg, y_train_reg, "neg_mean_absolute_error", "MAE",
         KFold(n_splits=lc["cv_folds"], shuffle=True, random_state=rs), -1),
    ]
    for ax, (task, model, y, scoring, label, cv, sign) in zip(axes, tasks):
        n, train_sc, val_sc = learning_curve(clone(model), X_train, y.squeeze(), train_sizes=sizes,
                                             cv=cv, scoring=scoring, n_jobs=1)
        train_sc, val_sc = sign * train_sc, sign * val_sc
        for i, size in enumerate(n):
            rows.append({"Tarea": task, "Metrica": label, "n_train": int(size),
                         "train_mean": train_sc[i].mean(), "train_std": train_sc[i].std(),
                         "val_mean": val_sc[i].mean(), "val_std": val_sc[i].std()})
        for scores, color, name in [(train_sc, "#BDC3C7", "Train"), (val_sc, "#1DB954", "Validación")]:
            ax.plot(n, scores.mean(axis=1), "o-", color=color, label=name)
            ax.fill_between(n, scores.mean(axis=1) - scores.std(axis=1),
                            scores.mean(axis=1) + scores.std(axis=1), color=color, alpha=0.2)
        ax.set_xlabel("Canciones de entrenamiento")
        ax.set_ylabel(label)
        titles = {"clasificacion": "Clasificación (F1, mayor es mejor)", "regresion": "Regresión (MAE, menor es mejor)"}
        ax.set_title(f"{titles[task]} — Gradient Boosting")
        ax.legend()

    fig.suptitle("Curvas de aprendizaje (validación cruzada de 3 folds)")
    fig.tight_layout()
    plt.close(fig)
    result = pd.DataFrame(rows).round(4)
    logger.info("Curvas de aprendizaje:\n%s", result.to_string(index=False))
    return result, fig
