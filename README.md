# Quantification Algo

Projet de recherche et de prototypage pour l'analyse de redondance fonctionnelle et algorithmique dans les modèles de langage et les architectures de type MoE.

## Objectif

Développer un moteur capable de détecter :
- redondances numériques, structurales et représentatives,
- sous-réseaux fonctionnellement équivalents,
- chemins de calcul algorithmiquement interchangeables,
- représentations partagées ou compressées avec un coût réel inférieur.

## Vision

Le projet part de l'hypothèse que la même information peut être encodée plusieurs fois, à plusieurs endroits, sous des formes différentes. L'objectif n'est pas seulement de compresser les poids, mais de comprendre les calculs et la signification de l'information qu'ils représentent.

## Structure du dépôt

```text
quantification-algo/
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── setup.sh
│   ├── check_environment.py
│   ├── run_demo.py
│   └── run_cpu_check.py
├── fid/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── tensor_inventory.py
│   │   └── graph_extractor.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── weight_fingerprint.py
│   │   ├── functional_similarity.py
│   │   └── redundancy_detector.py
│   └── reports/
│       └── analysis_report.py
├── tests/
│   └── test_tensor_inventory.py
└── examples/
    └── quickstart.py
```

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/GaTmaNnes/quantification-algo.git
cd quantification-algo
```

### 2. Créer un environnement virtuel et installer les dépendances

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ou via Makefile :

```bash
make setup
```

## Vérification de l'environnement

```bash
python scripts/check_environment.py
```

## Test CPU local

Le projet est conçu pour s'exécuter sur CPU pour les tests de base et les analyses de prototype.

```bash
python scripts/run_cpu_check.py
```

## Utilisation rapide

```bash
python -m fid --model sshleifer/tiny-gpt2 --device cpu --limit 10
```

Ou pour un prototype local minimal :

```bash
python scripts/run_demo.py
```

## Modules de base

### Core
- model_loader.py : charge un modèle Hugging Face réel ou un modèle local, puis construit l'inventaire des tenseurs
- tensor_inventory.py : collecte automatiquement les paramètres et buffers du modèle
- graph_extractor.py : extrait des informations de graphes de modules pour les analyses de structure

### Analysis
- weight_fingerprint.py : calcule des métriques de distribution et de sparsity sur les paramètres
- functional_similarity.py : compare des sorties ou représentations fonctionnelles
- redundancy_detector.py : détecte des modules ou paramètres similaires via des signatures statistiques

### Reports
- analysis_report.py : rend un rapport lisible en terminal

## Tests

```bash
pytest -q
```

## Roadmap

- [x] mise en place de la structure du dépôt
- [x] scripts d'installation et de validation
- [x] structure de base du package FID
- [x] chargement réel de modèles Hugging Face
- [x] extraction et inventaire des tenseurs
- [x] empreintes de poids et comparaison fonctionnelle
- [x] détection de redondance structurale / algorithmique basique
- [ ] validation expérimentale et rapports avancés

## Licence

Ce projet est distribué sous licence MIT.

## Remarque sur la visibilité

Le dépôt GitHub est actuellement privé. Pour le rendre public, il faut l'activer depuis l'interface GitHub :
- Settings
- General
- Visibility
- choisir Public

Cette modification ne peut pas être faite depuis cette session sans accès direct aux paramètres du dépôt GitHub.
