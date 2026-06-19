#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Script de lancement du mini projet Spam Classifier
#  Prérequis : Python 3.8 / 3.9, pip, Java 11+
# ─────────────────────────────────────────────────────────────
set -e

VENV_DIR=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Spam Classifier — Setup & Run"
echo "=============================================="

# ── 1. Venv ───────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Création de l'environnement virtuel …"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── 2. Dépendances ────────────────────────────────
echo "[setup] Installation des dépendances …"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ── 3. Java check (requis par PySpark) ────────────
if ! command -v java &>/dev/null; then
  echo "[ERREUR] Java non trouvé. Installez Java 11+ : sudo apt install openjdk-11-jdk"
  exit 1
fi
echo "[setup] Java : $(java -version 2>&1 | head -1)"

# ── 4. Lancement du pipeline ──────────────────────
echo ""
echo "[run] Démarrage du pipeline …"
python main.py

# ── 5. MLflow UI en arrière-plan ─────────────────
echo ""
echo "[mlflow] Démarrage de l'UI MLflow sur http://localhost:5000 …"
mlflow ui --port 5000 &
echo "  PID mlflow : $!"
echo "  Ouvrez http://localhost:5000 dans votre navigateur"
echo ""
echo "  Appuyez sur Ctrl+C pour arrêter."
wait
