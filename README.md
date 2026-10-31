# FID — Functional Information Deduplication

A small research prototype to inventory tensor parameters, compute simple weight fingerprints and compare modules for potential redundancy, with a first focus on Hugging Face models and MoE-style expert analysis.

## Résumé en français

FID (Functional Information Deduplication) est un prototype de recherche destiné à inventorier les paramètres d’un modèle, calculer des empreintes de poids légères et comparer des modules pour repérer des redondances potentielles. Le projet s’appuie sur des modèles Hugging Face et vise en priorité l’analyse de structures de type MoE (Mixture of Experts).

Le dépôt a pour objectif de fournir un socle de test pour :

- charger un modèle réel depuis Hugging Face sur CPU;
- lister automatiquement ses paramètres et ses buffers;
- calculer des empreintes rapides sur les tenseurs;
- détecter des modules similaires ou des motifs de type experts;
- exposer une interface CLI légère pour l’expérimentation locale.

### Structure du dépôt

- `fid/core/` : chargement des modèles et inventaire des tenseurs ;
- `fid/analysis/` : similarité fonctionnelle, empreintes de poids et détection de redondance ;
- `fid/reports/` : synthèse des résultats ;
- `scripts/` : vérification de l’environnement et scripts de démonstration ;
- `tests/` : tests de validation rapide ;
- `examples/` : exemple d’utilisation Python.

### État actuel du projet

Le dépôt est un prototype de recherche orienté analyse de modèles et détection de similarité. Il ne vise pas encore une optimisation complète de MoE ni une reconstruction systématique de modèles, mais il fournit une base exploitable pour l’exploration de tenseurs, l’analyse de redondance et la comparaison de modules.

Le README ci-dessous contient à la fois une documentation technique rapide et un cahier des charges scientifique plus complet. Les sections de recherche et les références bibliographiques sont conservées pour documenter la direction de recherche, sans remplacer les usages concrets du code.

## Goals

- Load a real model from Hugging Face on CPU.
- Inventory parameters and buffers automatically.
- Compute lightweight fingerprints across tensors.
- Detect similar modules and expert-like patterns.
- Expose a fast CLI for local experimentation.
