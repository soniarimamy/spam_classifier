#!/bin/bash
set -e

echo "============================================================"
echo "  Démarrage du pipeline Spam Classifier"
echo "============================================================"

python main.py

echo ""
echo "============================================================"
echo "  Pipeline terminé — démarrage des services web"
echo "============================================================"

# MLflow UI en arrière-plan
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:///app/mlruns &

echo "  MLflow UI      → http://localhost:5000"
echo "  Rapport        → http://localhost:8080/spam_classifier_report.html"
echo "============================================================"

# Serveur HTTP pour le rapport EvidentlyAI (maintient le conteneur actif)
python -m http.server 8080 --directory /app/reports
