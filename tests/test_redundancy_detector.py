# Quantification Algo

Projet de recherche et de prototypage pour l'analyse de redondance fonctionnelle et algorithmique dans les modèles de langage et les architectures de type MoE.

## Objectif

Développer un moteur capable de détecter :
- redondances numériques,
- redondances structurelles,
- redondances représentationnelles,
- sous-réseaux ou chemins de calcul fonctionnellement équivalents,
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
│   ├── test_tensor_inventory.py
│   └── test_redundancy_detector.py
└── examples/
    └── quickstart.py
```

## Installation

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

## Vérification rapide

```bash
python scripts/check_environment.py
```

## Test CPU local

```bash
python scripts/run_cpu_check.py
```

## Analyse réelle sur un modèle Hugging Face

```bash
python -m fid --model sshleifer/tiny-gpt2 --device cpu --limit 10 --top-similar 5
```

## Modules de base

### Core
- model_loader.py : charge un modèle Hugging Face réel et construit l'inventaire des tenseurs
- tensor_inventory.py : collecte automatiquement les paramètres et buffers du modèle
- graph_extractor.py : extrait des informations de structures de modules

### Analysis
- weight_fingerprint.py : calcule des statistiques de poids utiles pour des comparaisons de modules
- functional_similarity.py : compare des sorties / représentations fonctionnelles
- redundancy_detector.py : détecte des signatures de modules voisines et des candidats MoE comparables

### Reports
- analysis_report.py : produit un rapport terminal lisible

## Tests

```bash
pytest -q
```

## Roadmap

- [x] structure du dépôt
- [x] environnement et scripts
- [x] modèle Hugging Face réel
- [x] inventaire de tenseurs
- [x] empreintes de poids
- [x] détection de similarité de modules
- [x] rapport terminal
- [ ] analyse d'activations et validation fonctionnelle avancée

## Licence

MIT
