# Spam Classifier — PySpark + MLflow + EvidentlyAI

Pipeline complet de classification spam/ham entraîné sur des données synthétiques,
avec prétraitement distribué, sélection automatique de modèle, suivi d'expériences et dashboard de monitoring.

---

## Architecture du pipeline

```
generate_data.py       preprocess.py          train.py               evaluate.py
─────────────────   →  ─────────────────   →  ─────────────────   →  ─────────────────
300 messages           PySpark                Cross-validation       EvidentlyAI
synthétiques           Tokenisation           SGD / SVC / PA         ClassificationPreset
(150 spam/ham)         TF-IDF (500 dims)      MLflow tracking        DataDriftPreset
data/messages.csv      numpy array            meilleur modèle        reports/*.html
```

---

## Dashboards

| Dashboard | URL | Description |
|-----------|-----|-------------|
| EvidentlyAI | http://localhost:8080/spam_classifier_report.html | Métriques classification + data drift |
| MLflow UI | http://localhost:5000 | Historique des expériences et artefacts |

> Les dashboards s'activent une fois le pipeline terminé et **restent accessibles en permanence**.

---

## 3 façons de lancer le projet

### Option 1 — Sans Docker (local)

**Prérequis :** Python 3.9, Java 11+

```bash
# Créer l'environnement virtuel
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Lancer le pipeline
python main.py

# Dans un second terminal, lancer MLflow UI
mlflow ui
```

Puis ouvrir dans le navigateur :
- http://localhost:5000
- http://localhost:8080

---

### Option 2 — Docker Compose (build local)

**Prérequis :** Docker ≥ 24, Docker Compose ≥ 2.20

```bash
docker compose up --build -d
```

Suivre la progression :

```bash
docker compose logs -f
```

Puis ouvrir dans le navigateur :
- http://localhost:5000
- http://localhost:8080/spam_classifier_report.html

---

### Option 3 — Télécharger l'image depuis Docker Hub

**Prérequis :** Docker ≥ 24, Docker Compose ≥ 2.20

```bash
# 1. Se connecter à Docker Hub
docker login -u your_dockerhub_username

# 2. Télécharger l'image
docker pull rochel05/spam_classifier-spam-classifier:latest

# 3. Démarrer le conteneur
docker compose up -d
```

Puis ouvrir dans le navigateur :
- http://localhost:5000
- http://localhost:8080/spam_classifier_report.html

Pour arrêter :

```bash
docker compose down
```

---

## Structure du projet

```
spam_classifier/
├── main.py              # Orchestrateur — exécute les 4 étapes
├── generate_data.py     # Génère 300 messages synthétiques spam/ham
├── preprocess.py        # PySpark : tokenisation + TF-IDF (500 features)
├── train.py             # Cross-validation + tracking MLflow
├── evaluate.py          # Dashboard EvidentlyAI (HTML)
├── requirements.txt     # Dépendances Python
├── Dockerfile           # Image Docker (Python 3.9 + Java 11)
├── docker-compose.yml   # Service : pipeline + MLflow UI + serveur rapport
├── entrypoint.sh        # Script de démarrage du conteneur
├── data/                # Données générées à l'exécution
└── reports/             # Rapports HTML générés à l'exécution
```

---

## Paramètres configurables

| Fichier | Variable | Valeur par défaut | Description |
|---------|----------|-------------------|-------------|
| `main.py` | `n_samples` | `300` | Nombre de messages générés |
| `train.py` | `TEST_SIZE` | `0.25` | Proportion jeu de test |
| `preprocess.py` | `NUM_FEATURES` | `500` | Dimensions TF-IDF |

---

## Stack technique

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Prétraitement | PySpark 3.5 | Tokenisation distribuée + TF-IDF |
| Sélection modèle | scikit-learn 1.1 | Cross-validation SGD / LinearSVC / PassiveAggressive |
| Suivi | MLflow 2.9 | Logging métriques, artefacts, modèles |
| Monitoring | EvidentlyAI 0.4 | Dashboard performance + data drift |
| Image Docker | rochel05/spam_classifier-spam-classifier | Docker Hub |
