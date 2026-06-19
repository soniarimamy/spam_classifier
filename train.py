"""
Entraînement avec auto-sklearn (Logistic Regression via SGD)
+ tracking complet dans MLflow.
"""
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import autosklearn.classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix,
)

MLFLOW_EXPERIMENT = "spam-classifier"
TIME_BUDGET_SEC   = 120   # durée max auto-sklearn (augmenter pour de meilleurs résultats)
TEST_SIZE         = 0.25
RANDOM_STATE      = 42


def train(X: np.ndarray, y: np.ndarray):
    """
    Divise les données, lance auto-sklearn contraint à SGD (≈ Logistic Regression),
    logue tout dans MLflow, renvoie (model, X_test, y_test, y_pred, y_proba).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[train] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="autosklearn-logreg") as run:
        print(f"[train] MLflow run id : {run.info.run_id}")

        # ── auto-sklearn : restreint à SGD (loss=log ≈ Logistic Regression) ──
        print(f"[train] Lancement auto-sklearn (budget={TIME_BUDGET_SEC}s) …")
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=TIME_BUDGET_SEC,
            per_run_time_limit=30,
            seed=RANDOM_STATE,
            memory_limit=3072,
            # Restreindre aux classifieurs linéaires proches de la régression logistique
            include={"classifier": ["sgd", "liblinear_svc", "passive_aggressive"]},
            metric=autosklearn.metrics.f1,
        )
        automl.fit(X_train, y_train)
        print("[train] Modèles explorés par auto-sklearn :")
        print(automl.leaderboard())

        # ── Prédictions ───────────────────────────────────────────────────────
        y_pred  = automl.predict(X_test)
        y_proba = automl.predict_proba(X_test)[:, 1]

        # ── Métriques ─────────────────────────────────────────────────────────
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba)

        print("\n[train] ── Résultats ──────────────────────────────")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

        # ── Log MLflow ────────────────────────────────────────────────────────
        mlflow.log_params({
            "time_budget_sec": TIME_BUDGET_SEC,
            "test_size":       TEST_SIZE,
            "random_state":    RANDOM_STATE,
            "classifiers":     "sgd,liblinear_svc,passive_aggressive",
        })
        mlflow.log_metrics({
            "accuracy":  acc,
            "f1_score":  f1,
            "precision": prec,
            "recall":    rec,
            "roc_auc":   auc,
        })

        # Matrice de confusion comme artefact
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["ham", "spam"], columns=["pred_ham", "pred_spam"])
        os.makedirs("reports", exist_ok=True)
        cm_path = "reports/confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path)

        # Modèle auto-sklearn loggué dans MLflow
        mlflow.sklearn.log_model(automl, artifact_path="model")
        print(f"[train] Modèle sauvegardé dans MLflow → runs:/{run.info.run_id}/model")

    return automl, X_test, y_test, y_pred, y_proba
