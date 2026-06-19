# Spam Classifier — PySpark + auto-sklearn + MLflow + EvidentlyAI

Pipeline complet de classification spam/ham entraîné sur des données synthétiques,
avec prétraitement distribué, AutoML, suivi d'expériences et dashboard de monitoring.

---

## Architecture du pipeline

```
generate_data.py       preprocess.py          train.py               evaluate.py
─────────────────   →  ─────────────────   →  ─────────────────   →  ─────────────────
300 messages           PySpark                auto-sklearn           EvidentlyAI
synthétiques           Tokenisation           120 s de budget        ClassificationPreset
(150 spam/ham)         TF-IDF (500 dims)      SGD / SVC / PA         DataDriftPreset
data/messages.csv      numpy array            MLflow tracking        reports/*.html
```

---

## Démarrage rapide — Docker (recommandé)

### Prérequis
- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/install/) ≥ 2.20

### Lancer le projet

```bash
docker compose up --build -d
```

Le pipeline s'exécute automatiquement (~3 minutes à cause du budget auto-sklearn).
Suivre les logs :

```bash
docker compose logs -f
```

### Accéder aux dashboards

| Dashboard | URL | Description |
|-----------|-----|-------------|
| EvidentlyAI | http://localhost:8080/spam_classifier_report.html | Métriques classification + data drift |
| MLflow UI | http://localhost:5000 | Historique des expériences et artefacts |

> **Note :** Les dashboards ne sont disponibles qu'une fois le pipeline terminé
> (message `Pipeline terminé !` dans les logs).

### Arrêter

```bash
docker compose down
```

---

## Démarrage local (sans Docker)

### Prérequis
- Python 3.9
- Java 11+ (`java -version`)

```bash
# Créer l'environnement
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Lancer le pipeline
python main.py

# Ouvrir l'interface MLflow (dans un autre terminal)
mlflow ui        # → http://localhost:5000
```

Le rapport EvidentlyAI s'ouvre automatiquement dans le navigateur par défaut.

---

## Structure du projet

```
spam_classifier/
├── main.py              # Orchestrateur — exécute les 4 étapes
├── generate_data.py     # Génère 300 messages synthétiques spam/ham
├── preprocess.py        # PySpark : tokenisation + TF-IDF (500 features)
├── train.py             # auto-sklearn + tracking MLflow
├── evaluate.py          # Dashboard EvidentlyAI (HTML)
├── requirements.txt     # Dépendances Python
├── run.sh               # Script de lancement local avec venv
├── Dockerfile           # Image Docker (Python 3.9 + Java 11)
├── docker-compose.yml   # Services : pipeline + MLflow UI + serveur rapport
├── entrypoint.sh        # Script de démarrage du conteneur
├── data/                # Données générées à l'exécution
└── reports/             # Rapports HTML générés à l'exécution
```

---

## Paramètres configurables

| Fichier | Variable | Valeur par défaut | Description |
|---------|----------|-------------------|-------------|
| `main.py` | `n_samples` | `300` | Nombre de messages générés |
| `train.py` | `TIME_BUDGET_SEC` | `120` | Budget temps auto-sklearn (secondes) |
| `train.py` | `TEST_SIZE` | `0.25` | Proportion jeu de test |
| `preprocess.py` | `NUM_FEATURES` | `500` | Dimensions TF-IDF |

---

## Stack technique

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Prétraitement | PySpark 3.5 | Tokenisation distribuée + TF-IDF |
| AutoML | auto-sklearn 0.15 | Sélection automatique du modèle |
| Suivi | MLflow 2.9 | Logging métriques, artefacts, modèles |
| Monitoring | EvidentlyAI 0.4 | Dashboard performance + data drift |
