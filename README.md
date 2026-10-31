# Quantification Algo

Projet de recherche et de prototypage pour l'analyse de redondance fonctionnelle et algorithmiquedans les modèles de langages et architectures de type MoE.

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
│   └── run_demo.py
├── fid/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── tensor_inventory.py
│   └── analysis/
│       ├── __init__.py
│       ├── weight_fingerprint.py
│       ├── functional_similarity.py
│       └── redundancy_detector.py
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

## Utilisation rapide

```bash
python scripts/run_demo.py
```

Ou via le package FID :

```bash
python -m fid
```

## Modules de base

### Core
- model_loader.py : charge un modèle et expose les modules sélectionnés
- tensor_inventory.py : répertorie et décrit les tenseurs du modèle

### Analysis
- weight_fingerprint.py : calcule des empreintes de poids et métriques de structure
- functional_similarity.py : compare des sorties ou des représentations fonctionnelles
- redundancy_detector.py : détecte les éléments redondants ou équivalents

## Exemples

```bash
python examples/quickstart.py
```

## Roadmap

- [x] mise en place de la structure du dépôt
- [x] scripts d'installation et de validation
- [x] structure de base du package FID
- [ ] chargement de modèles Hugging Face
- [ ] extraction et inventaire des tenseurs
- [ ] empreintes de poids et comparaison fonctionnelle
- [ ] détection de redondance structurale / algorithmique
- [ ] validation expérimentale et rapports

## Licence

Ce projet est distribué sous licence MIT.

## Remarque sur la visibilité

Le dépôt GitHub est actuellement privé. Pour le rendre public, il faut l'activer depuis l'interface GitHub :
- Settings
- General
- Visibility
- choisir Public

Ce changement ne peut pas être effectué depuis cette session de tooling sans accès direct à la configuration du dépôt.
