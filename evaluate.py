"""
Génération du dashboard EvidentlyAI :
  - Rapport de performance de classification
  - Détection de data drift entre référence et données courantes
Ouvre automatiquement le rapport HTML dans le navigateur.
"""
import os
import webbrowser
import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset, DataDriftPreset
from evidently import ColumnMapping


REPORT_PATH = "reports/spam_classifier_report.html"


def build_evidently_dataframes(
    X_ref: np.ndarray, y_ref: np.ndarray, y_pred_ref: np.ndarray, y_proba_ref: np.ndarray,
    X_cur: np.ndarray, y_cur: np.ndarray, y_pred_cur: np.ndarray, y_proba_cur: np.ndarray,
) -> tuple:
    """Construit les DataFrames référence / courant pour EvidentlyAI."""
    feature_cols = [f"f{i}" for i in range(X_ref.shape[1])]

    def make_df(X, y, y_pred, y_proba):
        df = pd.DataFrame(X, columns=feature_cols)
        df["target"]          = y.astype(int)
        df["prediction"]      = y_pred.astype(int)
        df["prediction_proba"] = y_proba
        return df

    ref_df = make_df(X_ref, y_ref, y_pred_ref, y_proba_ref)
    cur_df = make_df(X_cur, y_cur, y_pred_cur, y_proba_cur)
    return ref_df, cur_df, feature_cols


def generate_report(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    y_pred: np.ndarray,  y_proba: np.ndarray,
    model,
):
    """
    Génère le rapport EvidentlyAI et l'ouvre dans le navigateur.
    - référence  = prédictions sur le train set
    - courant    = prédictions sur le test set
    """
    print("[evaluate] Construction des DataFrames EvidentlyAI …")

    # Prédictions sur le train (référence)
    y_pred_train  = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]

    ref_df, cur_df, feature_cols = build_evidently_dataframes(
        X_train, y_train, y_pred_train, y_proba_train,
        X_test,  y_test,  y_pred,       y_proba,
    )

    # ── ColumnMapping ─────────────────────────────────────────────────────────
    column_mapping = ColumnMapping(
        target          = "target",
        prediction      = "prediction",
        pos_label       = 1,
        numerical_features = feature_cols,
    )

    # ── Rapport EvidentlyAI ───────────────────────────────────────────────────
    print("[evaluate] Génération du rapport EvidentlyAI …")
    report = Report(metrics=[
        ClassificationPreset(),   # métriques classification : accuracy, F1, ROC, matrice de confusion
        DataDriftPreset(          # détection de drift entre train et test
            num_stattest="ks",
            cat_stattest="chisquare",
            num_stattest_threshold=0.05,
        ),
    ])

    report.run(
        reference_data = ref_df,
        current_data   = cur_df,
        column_mapping = column_mapping,
    )

    os.makedirs("reports", exist_ok=True)
    report.save_html(REPORT_PATH)
    print(f"[evaluate] Rapport sauvegardé → {REPORT_PATH}")

    # Ouvre le rapport dans le navigateur (désactivé en environnement Docker)
    if not os.environ.get("DOCKER"):
        abs_path = os.path.abspath(REPORT_PATH)
        webbrowser.open(f"file://{abs_path}")
        print(f"[evaluate] Dashboard ouvert : file://{abs_path}")
    else:
        print(f"[evaluate] Docker détecté — ouvrez http://localhost:8080/spam_classifier_report.html")
