"""
Entraînement avec sklearn (SGD / LinearSVC / PassiveAggressive)
sélection automatique par cross-validation F1 + tracking MLflow.
Remplace auto-sklearn pour résoudre les conflits de dépendances Docker.
"""
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix,
)

MLFLOW_EXPERIMENT = "spam-classifier"
TEST_SIZE         = 0.25
RANDOM_STATE      = 42

CANDIDATES = {
    "sgd": SGDClassifier(
        loss="log_loss", max_iter=1000, random_state=RANDOM_STATE
    ),
    "liblinear_svc": CalibratedClassifierCV(
        LinearSVC(max_iter=2000, random_state=RANDOM_STATE)
    ),
    "passive_aggressive": CalibratedClassifierCV(
        PassiveAggressiveClassifier(max_iter=1000, random_state=RANDOM_STATE)
    ),
}


def train(X: np.ndarray, y: np.ndarray):
    """
    Divise les données, évalue les 3 classifieurs par CV (F1),
    entraîne le meilleur, logue tout dans MLflow.
    Renvoie (model, X_test, y_test, y_pred, y_proba).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[train] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── Sélection du meilleur classifieur par cross-validation ────────────────
    print("[train] Évaluation des classifieurs (CV 3-fold, F1) …")
    leaderboard = {}
    for name, clf in CANDIDATES.items():
        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="f1", n_jobs=-1)
        leaderboard[name] = scores.mean()
        print(f"  {name:<22} F1 = {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(leaderboard, key=leaderboard.get)
    best_clf  = CANDIDATES[best_name]
    print(f"[train] Meilleur classifieur : {best_name} (F1={leaderboard[best_name]:.4f})")

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"sklearn-{best_name}") as run:
        print(f"[train] MLflow run id : {run.info.run_id}")

        best_clf.fit(X_train, y_train)

        # ── Prédictions ───────────────────────────────────────────────────────
        y_pred  = best_clf.predict(X_test)
        y_proba = best_clf.predict_proba(X_test)[:, 1]

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
            "best_classifier": best_name,
            "test_size":       TEST_SIZE,
            "random_state":    RANDOM_STATE,
            "classifiers":     "sgd,liblinear_svc,passive_aggressive",
            "cv_folds":        3,
        })
        mlflow.log_metrics({
            "accuracy":  acc,
            "f1_score":  f1,
            "precision": prec,
            "recall":    rec,
            "roc_auc":   auc,
            **{f"cv_f1_{n}": s for n, s in leaderboard.items()},
        })

        # Matrice de confusion comme artefact
        cm     = confusion_matrix(y_test, y_pred)
        cm_df  = pd.DataFrame(cm, index=["ham", "spam"], columns=["pred_ham", "pred_spam"])
        os.makedirs("reports", exist_ok=True)
        cm_path = "reports/confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path)

        mlflow.sklearn.log_model(best_clf, artifact_path="model")
        print(f"[train] Modèle sauvegardé dans MLflow → runs:/{run.info.run_id}/model")

    return best_clf, X_test, y_test, y_pred, y_proba
