"""
Point d'entrée du mini projet :

  Pipeline complet :
    1. generate_data  → données synthétiques spam/ham
    2. preprocess     → PySpark TF-IDF
    3. train          → auto-sklearn (Logistic Regression) + MLflow
    4. evaluate       → dashboard EvidentlyAI

  Usage :
    python main.py
    mlflow ui          # ouvre http://localhost:5000
"""
import os
import sys

# ── Imports locaux ────────────────────────────────────────────────────────────
from generate_data import generate_dataset
from preprocess    import preprocess
from train         import train
from evaluate      import generate_report


DATA_PATH = "data/messages.csv"


def main():
    print("=" * 60)
    print("  Mini Projet : Spam Classifier")
    print("  PySpark + auto-sklearn + MLflow + EvidentlyAI")
    print("=" * 60)

    # ── Étape 1 : Génération des données ─────────────────────────────────────
    print("\n[1/4] Génération des données …")
    generate_dataset(n_samples=300, output_path=DATA_PATH)

    # ── Étape 2 : Preprocessing PySpark (TF-IDF) ─────────────────────────────
    print("\n[2/4] Preprocessing PySpark (Tokenisation + TF-IDF) …")
    X, y, df_raw = preprocess(csv_path=DATA_PATH)

    # ── Étape 3 : Entraînement auto-sklearn + MLflow ──────────────────────────
    print("\n[3/4] Entraînement auto-sklearn + MLflow …")
    model, X_test, y_test, y_pred, y_proba = train(X, y)

    # ── Étape 4 : Dashboard EvidentlyAI ──────────────────────────────────────
    print("\n[4/4] Génération du dashboard EvidentlyAI …")
    # X_train = tout sauf le test set (reconstitution approximative)
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    generate_report(X_train, y_train, X_test, y_test, y_pred, y_proba, model)

    print("\n" + "=" * 60)
    print("  Pipeline terminé !")
    print(f"  Dashboard EvidentlyAI → reports/spam_classifier_report.html")
    print(f"  MLflow UI             → lancez : mlflow ui")
    print(f"                          puis ouvrez http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
