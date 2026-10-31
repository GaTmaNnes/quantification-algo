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




FUNCTIONAL INFORMATION DEDUPLICATION

Cahier des charges de recherche et dâ€™implementation

Version : 0.1
Date : 2026-10-31
Statut : Hypothese de recherche / prototype a construire

================================================================

1. OBJECTIF GENERAL
    ================================================================

Developper un moteur capable de decouvrir les redondances
dâ€™information et de calcul a lâ€™interieur dâ€™un LLM, meme lorsque
cette information est representee :

* A plusieurs endroits du modele.
* Avec des poids numeriquement differents.
* Sous des formes mathematiques differentes.
* Par plusieurs combinaisons de tenseurs.
* Par des sous-reseaux ou algorithmes differents.
* Dans des couches non adjacentes.
* Dans plusieurs experts dâ€™un reseau MoE.

Lâ€™objectif nâ€™est pas seulement de supprimer des poids similaires.

Il sâ€™agit de determiner si plusieurs chemins de calcul :

1. Representent une information equivalente.
2. Produisent des sorties fonctionnellement equivalentes.
3. Peuvent etre remplaces par une representation commune.
4. Peuvent etre executes a un cout inferieur.
5. Conservent les capacites du modele apres reconstruction.

Hypothese centrale :

UNE MEME INFORMATION PEUT ETRE ENCODEE PLUSIEURS FOIS,
A PLUSIEURS ENDROITS, SOUS DES FORMES NUMERIQUES,
MATHEMATIQUES ET ALGORITHMIQUES DIFFERENTES.

La similarite des poids nâ€™est donc pas une condition necessaire
pour identifier une redondance fonctionnelle.

================================================================
2. DISTINCTIONS FONDAMENTALES

Il faut separer cinq niveaux de redondance.

NIVEAU 1 â€” REDONDANCE NUMERIQUE

* Poids identiques ou proches.
* Blocs de tenseurs repetitifs.
* Structures de matrices similaires.

NIVEAU 2 â€” REDONDANCE STRUCTURELLE

* Matrices ayant des sous-espaces communs.
* Directions dominantes similaires.
* Rang effectif reduit.
* Facteurs partageables.

NIVEAU 3 â€” REDONDANCE REPRESENTATIONNELLE

* Poids differents.
* Activations internes similaires.
* Informations representees sous des formes differentes.

NIVEAU 4 â€” EQUIVALENCE FONCTIONNELLE

* Architectures differentes.
* Parametres differents.
* Sorties identiques ou proches sur un domaine donne.

NIVEAU 5 â€” EQUIVALENCE ALGORITHMIQUE

* Chemins de calcul differents.
* Operations mathematiques differentes.
* Meme fonction finale ou approximation suffisamment precise.
* Possibilite de remplacer plusieurs calculs par un seul.

Le niveau 5 constitue une piste de recherche prioritaire.

ATTENTION :
Une equivalence mesuree sur un jeu de calibration ne constitue
pas une preuve dâ€™equivalence universelle.

================================================================
3. PUBLICATIONS ET METHODES A ETUDIER

A. FUSION DE COUCHES

1. FCM â€” Fusion de couches non adjacentes
    GitHub :
    https://github.com/Usama1002/fcm-layer-fusion

ArXiv :
https://arxiv.org/abs/2610.15159

Objectif :

* Rechercher les couches redondantes non adjacentes.
* Identifier les ressemblances fonctionnelles globales.
* Eviter de limiter la recherche aux couches voisines.

2. XMerge â€” Selection et reconstruction des couches

ArXiv :
https://arxiv.org/abs/2609.02083

Objectif :

* Selection des couches.
* Reconstruction apres suppression ou fusion.
* Evaluation des compromis entre profondeur et qualite.

3. SWM â€” Sliding-Window Merging

Publication :
https://ojs.aaai.org/index.php/AAAI/article/view/

Reference :
https://globalresearchspace.com/space?star=W4416043746

Objectif :

* Fusion locale de couches.
* Mesure de la redondance entre couches consecutives.
* Baseline de comparaison pour FCM et XMerge.

NOTE :
Verifier le lien exact de lâ€™article SWM et son identifiant
bibliographique avant integration dans le registre.

B. COMPRESSION TENSORIELLE ET REDONDANCE MOE

4. TD-MoE â€” Decomposition tensorielle des experts

GitHub :
https://github.com/ust-xu/TD-MoE

ICLR 2026 :
https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc8ee7c7ab5b5f6b1615045dfb617ed6-Abstract-Conference.html

Objectif :

* Identifier les facteurs communs entre experts.
* Factoriser les representations.
* Reduire le stockage des matrices.
* Etudier le partage de parametres.

5. MixT â€” Tensor Mixture

ArXiv :
https://arxiv.org/abs/2605.25344

Objectif :

* Compression tensorielle.
* Reduction du nombre de parametres.
* Reduction des FLOPs et de la memoire.
* Etude des projections denses.

6. MoNE â€” Remplacement dâ€™experts redondants

GitHub :
https://github.com/zxgx/mode-pd

ICLR 2026 :
https://proceedings.iclr.cc/paper_files/paper/2026/hash/431c7c0b59aeafa2182b4d6617884b53-Abstract-Conference.html

Objectif :

* Identifier les experts redondants.
* Remplacer certains experts.
* Conserver les specialisations importantes.

7. Panorama des methodes de fusion

GitHub :
https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications

Objectif :

* Bibliographie.
* Identification de methodes complementaires.
* Comparaison des familles de fusion.

C. AUTRES TRAVAUX A CROISER

8. Functional vs. Parametric Equivalence of ReLU Networks

ICLR :
https://iclr.cc/virtual_2020/poster_Bylx-TNKvH.html

ISTA :
https://www.ista.ac.at/en/publications/

Objectif :

* Comprendre comment des parametres differents peuvent
    representer exactement la meme fonction.
* Etudier les permutations de neurones.
* Etudier les changements dâ€™echelle compenses.

Limite :
Les resultats concernent des architectures ReLU sous
certaines hypotheses. Ne pas les generaliser directement
aux Transformers.

9. Similarity of Neural Network Models:
    A Survey of Functional and Representational Measures

DOI :
https://doi.org/10.1145/3728458

Objectif :

* Comparaison representationnelle.
* Comparaison fonctionnelle.
* Classification des mesures de similarite.
* Choix des metriques adaptees au probleme.

10. RED â€” Redundancy-based Neural Network Compression

NeurIPS :
https://papers.neurips.cc/paper_files/paper/2021/hash/ae5e3ce40e0404a45ecacaaf05e5f735-Abstract.html

Objectif :

* Analyse des distributions de poids.
* Detection de neurones redondants.
* Fusion de neurones.
* Compression structurelle.

11. Sub-MoE â€” Subspace Expert Merging

AAAI 2026 :
https://ojs.aaai.org/index.php/AAAI/article/view/39464

Objectif :

* Regroupement dâ€™experts.
* Identification de sous-espaces communs.
* Decomposition SVD partagee.
* Reduction du nombre dâ€™experts.

12. Global Rank and Sparsity Optimization

ICLR 2026 :
https://proceedings.iclr.cc/paper_files/paper/2026/hash/bb1e9f32181a8d6a834670d5b3e1c48d-Abstract-Conference.html

Objectif :

* Allocation globale des budgets de rang.
* Compression adaptative des couches.
* Combinaison faible rang et sparsity.

13. LoRP â€” Locality-Aware Redundancy Pruning

ACL :
https://aclanthology.org/2026.emnlp-main.1979/

Objectif :

* Comparaison des representations entre couches.
* Detection de redondance locale et globale.
* Regroupement des couches selon leur similarite.

14. PuzzleMoE

ICML :
https://proceedings.mlr.press/v306/zhao26ay.html

Objectif :

* Detection de redondance fine entre experts.
* Fusion selective des poids.
* Preservation des parametres specifiques.

15. FloE

ICML :
https://proceedings.mlr.press/v267/zhou25j.html

Objectif :

* Compression des matrices dâ€™experts.
* Prediction des experts necessaires.
* Reduction des mouvements entre memoires.

16. FIRM-MoE

AAAI :
https://ojs.aaai.org/index.php/AAAI/article/view/39106

Objectif :

* Decomposition fine des experts.
* Prediction multi-couche.
* Prechargement adaptatif.

17. SMoE

DOI :
https://doi.org/10.1109/ISCA66397.2026.00099

Objectif :

* Substitution dâ€™experts fonctionnellement proches.
* Exploitation du cache.
* Reduction des transferts inutiles.

18. MoBE â€” Mixture-of-Basis-Experts

ICLR 2026 :
https://proceedings.iclr.cc/paper_files/paper/2026/hash/431d53d513461ff155d5bc8faa9a440c-Abstract-Conference.html

Objectif :

* Bases partagees entre experts.
* Parametres specifiques aux experts.
* Reduction des representations redondantes.

IMPORTANT :
Toutes les publications, versions, metriques et URL doivent
etre reverifiees directement aupres des sources originales.
Les chiffres annonces par les auteurs ne sont pas des resultats
garantis sur dâ€™autres architectures.

================================================================
4. ARCHITECTURE LOGICIELLE FID

fid/
|
+â€“ core/
|   +â€“ model_loader.py
|   +â€“ graph_extractor.py
|   +â€“ tensor_inventory.py
|   +â€“ architecture_parser.py
|
+â€“ analysis/
|   +â€“ weight_fingerprint.py
|   +â€“ activation_capture.py
|   +â€“ functional_similarity.py
|   +â€“ equivalence_detector.py
|   +â€“ information_overlap.py
|   +â€“ subgraph_comparator.py
|   +â€“ algorithmic_equivalence.py
|
+â€“ transforms/
|   +â€“ factorization.py
|   +â€“ shared_basis.py
|   +â€“ tensor_decomposition.py
|   +â€“ block_decomposition.py
|   +â€“ subgraph_fusion.py
|   +â€“ algebraic_rewrite.py
|   +â€“ reconstruction.py
|
+â€“ profiler/
|   +â€“ flops.py
|   +â€“ memory.py
|   +â€“ transfer.py
|   +â€“ latency.py
|   +â€“ cache.py
|   +â€“ hardware_backend.py
|
+â€“ validation/
|   +â€“ numerical.py
|   +â€“ perplexity.py
|   +â€“ regression.py
|   +â€“ domain_coverage.py
|   +â€“ robustness.py
|
+â€“ reports/
|   +â€“ similarity_report.py
|   +â€“ equivalence_report.py
|   +â€“ compression_report.py
|   +â€“ hardware_report.py
|
+â€“ fid.py

================================================================
5. LES QUATRE MOTEURS Dâ€™ANALYSE

A. WEIGHT INFORMATION PROFILER

Analyser chaque tenseur :

* Shape.
* Dtype.
* Distribution.
* Moyenne et variance.
* Normes L1 et L2.
* Valeurs singulieres.
* Rang effectif.
* Sparsity.
* Correlation lignes/colonnes.
* Similarite entre blocs.
* Sous-espaces communs.
* Sensibilite a la quantification.

Metriques :

* SVD.
* QR.
* PCA.
* Random projections.
* Spectral analysis.
* Block similarity.

B. ACTIVATION ANALYZER

Capturer les activations intermediaires :

* Entree de chaque bloc.
* Sortie de chaque bloc.
* Entree et sortie de chaque expert.
* Activations gate/up/down des MLP.
* Sorties attention.
* Residual streams.
* Normalisations.

Metriques :

* CKA.
* Cosine similarity.
* Erreur relative.
* Correlation.
* Rang des representations.
* Distribution des activations.

C. FUNCTIONAL EQUIVALENCE ENGINE

Comparer deux fonctions :

f_A(x) et f_B(x).

Erreur relative :

E_rel =
||f_A(x)-f_B(x)||2 /
(||f_A(x)||2 + epsilon)

Mesurer :

* Erreur moyenne.
* Erreur maximale.
* Erreur par dimension.
* Erreur par categorie dâ€™entree.
* Erreur par contexte.
* Erreur sur sequences longues.
* Stabilite hors calibration.

D. ALGORITHMIC EQUIVALENCE ENGINE

Rechercher des chemins de calcul differents :

* Multiplication matricielle directe.
* Factorisation matricielle.
* Factorisation tensorielle.
* Decomposition par blocs.
* Bases partagees.
* Operations fusionnees.
* Reordonnancement algebrique.
* Remplacement de sous-graphes.

Exemple :

Y1 = X @ W

Y2 = (X @ A) @ B

Si W = A @ B, les deux chemins sont equivalentes
en arithmetique exacte.

Pour les approximations :

Y1 ~= Y2

Il faut mesurer lâ€™erreur numerique et fonctionnelle.

================================================================
6. MESURE DU COUT REEL

Ne jamais confondre compression des parametres et acceleration.

Pour chaque representation candidate, mesurer :

* Taille sur SSD.
* Taille en RAM.
* Taille en VRAM.
* Taille en memoire HTP.
* Taille des facteurs reconstruits.
* Nombre de FLOPs.
* Nombre de transferts.
* Temps SSD -&gt; RAM.
* Temps RAM -&gt; GPU.
* Temps RAM -&gt; HTP.
* Temps de reconstruction.
* Temps dâ€™attente du calculateur.
* Cache hit/miss.
* Latence prefill.
* Latence decode.
* Tokens par seconde.

Fonction de cout indicative :

C =
alpha * FLOPs

* beta * Memory
* gamma * Transfer
* delta * Latency

Les coefficients doivent etre calibres sur le materiel cible.

Conserver plusieurs solutions non dominees dans une
frontiere de Pareto.

================================================================
7. PIPELINE EXPERIMENTAL

ETAPE 1 â€” PROFILAGE INITIAL

* Charger le modele.
* Extraire les tenseurs.
* Construire le graphe.
* Capturer les activations.
* Produire les empreintes.

Aucun poids modifie.

ETAPE 2 â€” DETECTION

* Comparer les couches.
* Comparer les experts.
* Comparer les sous-graphes.
* Identifier les facteurs communs.
* Identifier les fonctions proches.
* Identifier les chemins algorithmiques equivalents.

ETAPE 3 â€” GENERATION

Generer des representations candidates :

* Low rank.
* Shared basis.
* Tensor factorization.
* Block decomposition.
* Algebraic rewrite.
* Subgraph fusion.
* Expert merging.

ETAPE 4 â€” EVALUATION

Pour chaque candidat :

* Erreur numerique.
* Erreur fonctionnelle.
* Perplexite.
* Parametres.
* RAM/VRAM/HTP.
* FLOPs.
* Latence.
* Transferts.

ETAPE 5 â€” RECONSTRUCTION

* Reconstruire le sous-reseau.
* Recalculer les dependances.
* Verifier les dimensions.
* Verifier le routage.
* Verifier les residuals et normalisations.
* Tester les cas limites.

ETAPE 6 â€” VALIDATION

* Comparer au modele original.
* Tester les jeux hors calibration.
* Tester les domaines rares.
* Tester les sequences longues.
* Comparer les performances reelles.
* Archiver les resultats et les empreintes SHA-256.

================================================================
8. EXPERIENCE PRIORITAIRE

EXPERIENCE FID-001 :

OBJECTIF :
Determiner si deux experts ayant des poids tres differents
peuvent etre representes par une structure partagee moins
couteuse sans degradation fonctionnelle importante.

PROTOCOLE :

1. Choisir deux experts dâ€™un modele MoE.
2. Recuperer toutes leurs matrices.
3. Capturer les entrees et sorties des deux experts.
4. Mesurer la similarite des poids.
5. Mesurer la similarite des activations.
6. Tester les sous-espaces communs.
7. Generer une factorisation partagee.
8. Comparer les sorties.
9. Mesurer les couts.
10. Tester sur un jeu hors calibration.

Candidats :

* Representation originale.
* Shared basis.
* SVD.
* Tensor factorization.
* Block decomposition.
* Representation algorithmique alternative.

RESULTATS ATTENDUS :

* Redondance detectee.
* Niveau de preuve.
* Erreur numerique.
* Gain memoire.
* Gain calcul.
* Cout de reconstruction.
* Impact sur la qualite.
* Comportement hors calibration.

================================================================
9. PRIORITE DES METHODES

PRIORITE 1 â€” DECOUVERTE DE REDONDANCE

* TD-MoE.
* FCM.
* PuzzleMoE.
* Sub-MoE.
* LoRP.

PRIORITE 2 â€” REPRESENTATION PARTAGEE

* MoBE.
* MixT.
* Global Rank/Sparsity.
* RED.

PRIORITE 3 â€” RECONSTRUCTION ET REDUCTION

* XMerge.
* SWM.
* MoNE.

PRIORITE 4 â€” OPTIMISATION DU RUNTIME

* FloE.
* FIRM-MoE.
* SMoE.

FID doit etre le moteur commun dâ€™analyse qui alimente ces
methodes, sans les confondre ni supposer leur compatibilite.

================================================================
10. OUTILS LOGICIELS

Python :
https://www.python.org/

PyTorch :
https://pytorch.org/

Hugging Face Transformers :
https://github.com/huggingface/transformers

PyTorch FX :
https://pytorch.org/docs/stable/fx.html

PyTorch Export :
https://pytorch.org/docs/stable/export.html

SciPy :
https://scipy.org/

NumPy :
https://numpy.org/

NetworkX :
https://networkx.org/

nnsight :
https://github.com/ndif-team/nnsight

GGUF :
https://github.com/ggml-org/ggml

llama.cpp :
https://github.com/ggml-org/llama.cpp

================================================================
11. REGLES DE VALIDATION SCIENTIFIQUE

Chaque constat doit avoir un statut :

OBSERVE :
Mesure directement obtenue.

DEMONTRÃ‰ :
Equivalence mathematiquement prouvee sous des hypotheses
explicites.

EXPERIMENTAL :
Resultat observe sur un jeu de donnees determine.

APPROCHE :
Similarite ou equivalence approximative.

HYPOTHESE :
Piste non encore validee.

INCONNU :
Information indisponible.

Ne jamais annoncer :

* Une equivalence universelle sur la base de quelques exemples.
* Une acceleration sur la seule base du nombre de parametres.
* Une reduction memoire sans mesure du pic de memoire.
* Une compatibilite GPU/HTP sans kernel fonctionnel.
* Un gain cumulatif en additionnant les chiffres de publications.

Conserver :

* Version du modele.
* Hash des poids.
* Configuration.
* Commit du code.
* Donnees de calibration.
* Donnees de validation.
* Resultats bruts.
* Logs.
* Metriques.
* Transformations appliquees.
* Limites connues.

================================================================
12. OBJECTIF FINAL

Construire un systeme capable de decouvrir automatiquement :

1. Quelles informations sont representees plusieurs fois.
2. Ou elles sont representees.
3. Sous quelles formes numeriques et mathematiques.
4. Quels sous-reseaux calculent des fonctions equivalentes.
5. Quels chemins algorithmiques sont interchangeables.
6. Quelle representation coute le moins cher.
7. Comment reconstruire le modele.
8. Comment valider quâ€™il conserve ses capacites.

VISION :

NE PLUS COMPRESSER UNIQUEMENT LES POIDS.

COMPRENDRE LES CALCULS ET LES INFORMATIONS QUâ€™ILS
REPRESENTENT, PUIS RECONSTRUIRE UNE ARCHITECTURE
QUI LES CALCULE AVEC LE MOINS DE RESSOURCES POSSIBLE.</pre>

======
FID - FUNCTIONAL INFORMATION DEDUPLICATION
RAPPORT DE VERIFICATION BIBLIOGRAPHIQUE
Version: 0.3
Date: 2026-10-31
Statut: CORRIGE - VALIDATION SCIENTIFIQUE PARTIELLE
================================================================

1. OBJECTIF
----------------------------------------------------------------

Completer FID avec les recherches portant sur:

- La redondance des representations internes.
- La reutilisation du flux residuel.
- La reconstruction du cache KV.
- La redondance des normalisations.
- La redondance temporelle des activations.
- La redondance entre branches de calcul.
- L'equivalence entre sous-graphes.
- La reecriture et l'optimisation des graphes.

Principe fondamental:

Deux composants peuvent etre fonctionnellement equivalents
sans avoir des poids similaires.

La similarite ne constitue pas une preuve d'equivalence.

================================================================
2. FLUX RESIDUEL ET REPRESENTATIONS INTERNES
================================================================

A. TRR - Token Representation Reuse

Publication:
https://doi.org/10.1007/s40747-026-02564-1

Resultats annonces:
- Jusqu'a 12.40 % de representations reutilisees.
- Tests sur Qwen2.5-1.5B-Instruct et Llama-3-8B.
- Resultats variables selon les benchmarks.
- Baisse observee sur BoolQ.
- Detection quadratique en longueur de sequence.

Limite:
Le cout de detection peut annuler le gain d'execution.

Interet FID:
- Comparaison des representations entre couches.
- Detection de representations reutilisables.
- Mesure du cout de detection.
- Validation hors distribution.

B. MLSAE - Multi-Layer Sparse Autoencoders

Titre:
Residual Stream Analysis with Multi-Layer SAEs

Paper:
https://arxiv.org/abs/2409.04185

OpenReview:
https://openreview.net/forum?id=XAjfjizaKs

Code:
https://github.com/tim-lawson/mlsae

Interet FID:
- Suivre les caracteristiques entre couches.
- Detecter les representations persistantes.
- Comparer les directions d'information.
- Etudier les changements de representation.

C. TALKING HEADS

Titre:
Talking Heads: Understanding Inter-Layer
Communication in Transformer Language Models

Paper:
https://proceedings.neurips.cc/paper_files/paper/2024/hash/70e5444e5f331f7f5431f302110b97af-Abstract-Conference.html

Interet FID:
- Etudier les communications entre couches.
- Identifier les sous-espaces communs.
- Comparer les canaux de communication.

D. DEPTH-ATTENTION

https://aclanthology.org/2026.emnlp-main.2085/

Interet:
Reutilisation selective de representations
produites par des couches precedentes.

E. DELTA ATTENTION RESIDUALS

https://arxiv.org/abs/2605.18855

Interet:
Comparer les contributions des couches plutot
que les seuls etats residuels cumulatifs.

================================================================
3. RECONSTRUCTION DU CACHE KV
================================================================

THE RESIDUAL STREAM IS ALL YOU NEED:
ON THE REDUNDANCY OF THE KV CACHE
IN TRANSFORMER INFERENCE

Paper:
https://arxiv.org/abs/2603.19664

Code:
https://github.com/Kaleemullahqasim/KV-Direct

Hugging Face:
https://huggingface.co/papers/2603.19664

Resultats annonces:
- Reconstruction du KV depuis le flux residuel.
- Tests sur six modeles de quatre familles.
- 5 Ko par token sur Gemma 3-4B.
- 136 Ko par token pour les paires KV dans
  la configuration comparee.
- Correspondance exacte des tokens en decodage
  glouton sur les modeles testes.

Interet FID:
- Identifier les representations reconstructibles.
- Mesurer la precision de reconstruction.
- Comparer stockage et recalcul.
- Mesurer la memoire maximale.
- Mesurer les couts de transfert.

Limite:
Ne pas generaliser automatiquement aux architectures
hybrides ou aux etats recurrents.

AUTRES REFERENCES:

https://github.com/facebookresearch/context-language-models/blob/main/suffix_cache_reuse/README.md

https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache.md

https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md

================================================================
4. REDONDANCE DES NORMALISATIONS
================================================================

A. LAYERNORM / RMSNORM

https://proceedings.neurips.cc/paper_files/paper/2023/hash/8f1bacee31caf990a4f08d84f0ccb322-Abstract-Conference.html

B. ENJOY YOUR LAYER NORMALIZATION

https://arxiv.org/abs/2605.14521

C. TAPERNORM

https://arxiv.org/abs/2602.10408

Pistes FID:
- Detecter les normalisations absorbables.
- Fusionner les transformations lineaires.
- Detecter les biais absorbables.
- Identifier les operations eliminables.
- Verifier les transformations algebriques.

Attention:
Une transformation exacte sous certaines conditions
n'est pas une equivalence universelle.

================================================================
5. REDONDANCE TEMPORELLE DES ACTIVATIONS
================================================================

A. TADP - TEMPORAL ACTIVATION DYNAMICS PRUNING

Paper:
https://aclanthology.org/2026.findings-emnlp.1837/

PDF:
https://aclanthology.org/2026.findings-emnlp.1837.pdf

B. COLA

Paper:
https://aclanthology.org/2025.emnlp-main.230/

Pistes FID:
- Mesurer les activations par token.
- Analyser les pics d'activation.
- Comparer les contributions temporelles.
- Detecter les canaux conditionnels.
- Mesurer la contribution reelle aux sorties.

Limite:
Un canal rarement actif peut rester indispensable
dans certains contextes.

================================================================
6. REDONDANCE DES POIDS
================================================================

WEIGHT-REDUNDANCY PRUNING

https://arxiv.org/abs/2609.09883

Utilisation FID:
- Detection initiale de blocs candidats.
- Comparaison des projections entre couches.
- Analyse des distributions de poids.
- Identification de similarites structurelles.

Limite:
La similarite des poids ne suffit pas a demontrer
une equivalence fonctionnelle.

================================================================
7. REDONDANCE DES EMBEDDINGS
================================================================

ON THE EFFECT OF (NEAR) DUPLICATE SUBWORDS
IN LANGUAGE MODELLING

https://aclanthology.org/2024.findings-acl.571/

Pistes FID:
- Comparer les lignes des matrices d'embedding.
- Detecter les directions communes.
- Etudier les variantes lexicales.
- Comparer embeddings d'entree et de sortie.
- Valider les effets sur les logits.

Limite:
Des embeddings proches ne sont pas necessairement
interchangeables dans tous les contextes.

================================================================
8. REDONDANCE ENTRE BRANCHES DE CALCUL
================================================================

SPECFOLD

Paper:
https://arxiv.org/abs/2610.04875

Hugging Face:
https://huggingface.co/papers/2610.04875

Pistes FID:
- Comparer les branches d'execution.
- Detecter les calculs communs.
- Reutiliser les etats caches.
- Identifier les divergences.
- Eviter les recalculs des parties communes.

Limite:
Travail sur le decodage speculatif des modeles
de langage par diffusion.
Ne pas generaliser directement aux LLM autoregressifs.

================================================================
9. COMPILATION ET REECRITURE DES GRAPHES
================================================================

LLM COMPILER OPTIMIZATIONS

https://github.com/khatami-mehrdad/llm-compiler-optimizations

Fonctions:
- Pipeline PyTorch vers MLIR.
- Reconnaissance des operations Transformer.
- Fusion de motifs.
- Separation prefill et decode.
- Optimisation de la representation intermediaire.

Limite:
La reduction du nombre de lignes IR ne signifie pas
une reduction equivalente des FLOPs ou de la latence.

Utilisation FID:
- Export du graphe.
- Detection des motifs.
- Generation des transformations.
- Verification.
- Compilation.
- Mesure sur materiel reel.

================================================================
10. MODULES A AJOUTER A FID
================================================================

fid/
|
+-- analysis/
|   +-- residual_stream.py
|   +-- hidden_state_reuse.py
|   +-- cross_layer_tracking.py
|   +-- temporal_activation.py
|   +-- token_embedding_overlap.py
|   +-- normalization_redundancy.py
|   +-- cross_operator_equivalence.py
|   +-- branch_redundancy.py
|   +-- conditional_redundancy.py
|   +-- kv_reconstruction.py
|
+-- transforms/
|   +-- residual_reconstruction.py
|   +-- representation_reuse.py
|   +-- norm_folding.py
|   +-- operator_rewrite.py
|   +-- cross_graph_replacement.py
|   +-- branch_computation_reuse.py
|
+-- compiler/
|   +-- graph_ir.py
|   +-- pattern_matcher.py
|   +-- graph_rewriter.py
|   +-- equivalence_verifier.py
|   +-- backend_dispatch.py
|
+-- profiler/
|   +-- detection_overhead.py
|   +-- reconstruction_cost.py
|   +-- execution_cost.py
|   +-- transfer_cost.py
|
+-- validation/
    +-- exact_equivalence.py
    +-- approximate_equivalence.py
    +-- conditional_validation.py
    +-- out_of_distribution.py
    +-- hardware_validation.py

================================================================
11. CLASSIFICATION DES PREUVES
================================================================

Chaque resultat FID doit utiliser une categorie:

OBSERVED_SIMILARITY
    Similarite observee.

REPRESENTATION_REUSE
    Representation reutilisable.

FUNCTIONAL_EQUIVALENCE
    Sorties equivalentes sur un domaine teste.

EXACT_ALGORITHMIC_EQUIVALENCE
    Transformation mathematiquement exacte.

APPROXIMATE_EQUIVALENCE
    Transformation avec erreur mesuree.

EXECUTION_SPEEDUP
    Acceleration mesuree sur materiel reel.

MEMORY_REDUCTION
    Reduction mesuree de la memoire maximale.

Ces categories ne sont pas interchangeables.

================================================================
12. EXPERIENCES PRIORITAIRES
================================================================

FID-001:
Comparer les representations cachees entre couches.

FID-002:
Tester la reconstruction KV depuis le flux residuel.

FID-003:
Detecter et fusionner les normalisations absorbables.

FID-004:
Decouvrir des sous-graphes fonctionnellement equivalents.

FID-005:
Etudier la redondance temporelle des canaux.

FID-006:
Compiler et executer les graphes transformes.

FID-007:
Mesurer le cout de detection et de reconstruction.

================================================================
13. LACUNES SCIENTIFIQUES
================================================================

LACUNE A:
Decouverte automatique d'equivalences entre
sous-graphes de structures differentes.

LACUNE B:
Equivalence entre familles d'operations differentes.

LACUNE C:
Redondance conditionnelle selon le token,
le contexte, la tache et la profondeur.

LACUNE D:
Distinction entre information partagee,
sorties similaires et calculs substituables.

LACUNE E:
Mesure du gain reel apres reconstruction,
compilation et execution.

================================================================
14. CONCLUSION
================================================================

FID doit depasser la compression classique.

Objectifs:

1. Detecter les informations dupliquees.
2. Identifier les representations interchangeables.
3. Detecter les sous-espaces communs.
4. Identifier les calculs redondants.
5. Decouvrir les sous-graphes equivalents.
6. Construire des transformations moins couteuses.
7. Verifier les transformations.
8. Mesurer les gains reels.

PIPELINE:

DETECTION
    |
ANALYSE FONCTIONNELLE
    |
DECOUVERTE D'EQUIVALENCE
    |
TRANSFORMATION
    |
VALIDATION
    |
COMPILATION
    |
EXECUTION OPTIMISEE

PRINCIPE FINAL:

Une similarite observee n'est pas une preuve
d'equivalence.

Une reduction theorique des FLOPs n'est pas
une preuve d'acceleration.

Une reduction de la taille des poids n'est pas
une preuve de reduction de la memoire maximale.

================================================================
15. VERDICT
================================================================

Statut: ACCEPT AVEC CORRECTIONS

Priorites scientifiques:

1. KV-Direct: reconstruction des representations.
2. TRR: reutilisation des representations.
3. Normalization folding: equivalence algebrique.
4. Cross-operator equivalence: recherche fondamentale.
5. Graph rewriting: exploitation pratique.

La principale differenciation de FID reste la
decouverte automatique d'equivalences fonctionnelles
et algorithmiques entre sous-graphes differents.

FIN DU RAPPORT

