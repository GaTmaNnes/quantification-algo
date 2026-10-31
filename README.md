# FID — Functional Information Deduplication

A small research prototype to inventory tensor parameters, compute simple weight fingerprints and compare modules for potential redundancy, with a first focus on Hugging Face models and MoE-style experts.

## Goals

- Load a real model from Hugging Face on CPU.
- Inventory parameters and buffers automatically.
- Compute lightweight fingerprints across tensors.
- Detect similar modules and expert-like patterns.
- Expose a fast CLI for local experimentation.

## Project structure

```text
quantification-algo/
├── fid/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── tensor_inventory.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── functional_similarity.py
│   │   ├── redundancy_detector.py
│   │   └── weight_fingerprint.py
│   └── reports/
│       └── analysis_report.py
├── scripts/
│   ├── check_environment.py
│   ├── run_cpu_check.py
│   ├── run_demo.py
│   └── setup.sh
├── tests/
│   ├── test_tensor_inventory.py
│   └── test_redundancy_detector.py
├── examples/
│   └── quickstart.py
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick validation

```bash
python scripts/check_environment.py
pytest -q
```

## Real model analysis

```bash
python -m fid --model sshleifer/tiny-gpt2 --device cpu --limit 10 --top-similar 5
```

This loads a tiny GPT-2 model from Hugging Face, inventories its tensors and prints a short redundancy summary.

## Example usage in Python

```python
from fid.core.model_loader import ModelLoader
from fid.analysis.redundancy_detector import RedundancyDetector

loader = ModelLoader("sshleifer/tiny-gpt2", device="cpu")
spec = loader.load()

detector = RedundancyDetector()
print(spec.inventory.count())
print(detector.fingerprints_from_model(spec.model)[0])
```

## Notes

This is intentionally incremental and research-oriented: the goal is not to perform a full MoE optimization yet, but to provide a robust base for tensor exploration, weight characterization and module comparison.




FID — FUNCTIONAL INFORMATION DEDUPLICATION

Cahier des charges de recherche et d’implementation

Version : 0.1
Date : 2026-10-31
Statut : Hypothese de recherche / prototype a construire

================================================================

1. OBJECTIF GENERAL
    ================================================================

Developper un moteur capable de decouvrir les redondances
d’information et de calcul a l’interieur d’un LLM, meme lorsque
cette information est representee :

* A plusieurs endroits du modele.
* Avec des poids numeriquement differents.
* Sous des formes mathematiques differentes.
* Par plusieurs combinaisons de tenseurs.
* Par des sous-reseaux ou algorithmes differents.
* Dans des couches non adjacentes.
* Dans plusieurs experts d’un reseau MoE.

L’objectif n’est pas seulement de supprimer des poids similaires.

Il s’agit de determiner si plusieurs chemins de calcul :

1. Representent une information equivalente.
2. Produisent des sorties fonctionnellement equivalentes.
3. Peuvent etre remplaces par une representation commune.
4. Peuvent etre executes a un cout inferieur.
5. Conservent les capacites du modele apres reconstruction.

Hypothese centrale :

UNE MEME INFORMATION PEUT ETRE ENCODEE PLUSIEURS FOIS,
A PLUSIEURS ENDROITS, SOUS DES FORMES NUMERIQUES,
MATHEMATIQUES ET ALGORITHMIQUES DIFFERENTES.

La similarite des poids n’est donc pas une condition necessaire
pour identifier une redondance fonctionnelle.

================================================================
2. DISTINCTIONS FONDAMENTALES

Il faut separer cinq niveaux de redondance.

NIVEAU 1 — REDONDANCE NUMERIQUE

* Poids identiques ou proches.
* Blocs de tenseurs repetitifs.
* Structures de matrices similaires.

NIVEAU 2 — REDONDANCE STRUCTURELLE

* Matrices ayant des sous-espaces communs.
* Directions dominantes similaires.
* Rang effectif reduit.
* Facteurs partageables.

NIVEAU 3 — REDONDANCE REPRESENTATIONNELLE

* Poids differents.
* Activations internes similaires.
* Informations representees sous des formes differentes.

NIVEAU 4 — EQUIVALENCE FONCTIONNELLE

* Architectures differentes.
* Parametres differents.
* Sorties identiques ou proches sur un domaine donne.

NIVEAU 5 — EQUIVALENCE ALGORITHMIQUE

* Chemins de calcul differents.
* Operations mathematiques differentes.
* Meme fonction finale ou approximation suffisamment precise.
* Possibilite de remplacer plusieurs calculs par un seul.

Le niveau 5 constitue une piste de recherche prioritaire.

ATTENTION :
Une equivalence mesuree sur un jeu de calibration ne constitue
pas une preuve d’equivalence universelle.

================================================================
3. PUBLICATIONS ET METHODES A ETUDIER

A. FUSION DE COUCHES

1. FCM — Fusion de couches non adjacentes
    GitHub :
    https://github.com/Usama1002/fcm-layer-fusion

ArXiv :
https://arxiv.org/abs/2610.15159

Objectif :

* Rechercher les couches redondantes non adjacentes.
* Identifier les ressemblances fonctionnelles globales.
* Eviter de limiter la recherche aux couches voisines.

2. XMerge — Selection et reconstruction des couches

ArXiv :
https://arxiv.org/abs/2609.02083

Objectif :

* Selection des couches.
* Reconstruction apres suppression ou fusion.
* Evaluation des compromis entre profondeur et qualite.

3. SWM — Sliding-Window Merging

Publication :
https://ojs.aaai.org/index.php/AAAI/article/view/

Reference :
https://globalresearchspace.com/space?star=W4416043746

Objectif :

* Fusion locale de couches.
* Mesure de la redondance entre couches consecutives.
* Baseline de comparaison pour FCM et XMerge.

NOTE :
Verifier le lien exact de l’article SWM et son identifiant
bibliographique avant integration dans le registre.

B. COMPRESSION TENSORIELLE ET REDONDANCE MOE

4. TD-MoE — Decomposition tensorielle des experts

GitHub :
https://github.com/ust-xu/TD-MoE

ICLR 2026 :
https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc8ee7c7ab5b5f6b1615045dfb617ed6-Abstract-Conference.html

Objectif :

* Identifier les facteurs communs entre experts.
* Factoriser les representations.
* Reduire le stockage des matrices.
* Etudier le partage de parametres.

5. MixT — Tensor Mixture

ArXiv :
https://arxiv.org/abs/2605.25344

Objectif :

* Compression tensorielle.
* Reduction du nombre de parametres.
* Reduction des FLOPs et de la memoire.
* Etude des projections denses.

6. MoNE — Remplacement d’experts redondants

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
* Etudier les changements d’echelle compenses.

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

10. RED — Redundancy-based Neural Network Compression

NeurIPS :
https://papers.neurips.cc/paper_files/paper/2021/hash/ae5e3ce40e0404a45ecacaaf05e5f735-Abstract.html

Objectif :

* Analyse des distributions de poids.
* Detection de neurones redondants.
* Fusion de neurones.
* Compression structurelle.

11. Sub-MoE — Subspace Expert Merging

AAAI 2026 :
https://ojs.aaai.org/index.php/AAAI/article/view/39464

Objectif :

* Regroupement d’experts.
* Identification de sous-espaces communs.
* Decomposition SVD partagee.
* Reduction du nombre d’experts.

12. Global Rank and Sparsity Optimization

ICLR 2026 :
https://proceedings.iclr.cc/paper_files/paper/2026/hash/bb1e9f32181a8d6a834670d5b3e1c48d-Abstract-Conference.html

Objectif :

* Allocation globale des budgets de rang.
* Compression adaptative des couches.
* Combinaison faible rang et sparsity.

13. LoRP — Locality-Aware Redundancy Pruning

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

* Compression des matrices d’experts.
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

* Substitution d’experts fonctionnellement proches.
* Exploitation du cache.
* Reduction des transferts inutiles.

18. MoBE — Mixture-of-Basis-Experts

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
garantis sur d’autres architectures.

================================================================
4. ARCHITECTURE LOGICIELLE FID

fid/
|
+– core/
|   +– model_loader.py
|   +– graph_extractor.py
|   +– tensor_inventory.py
|   +– architecture_parser.py
|
+– analysis/
|   +– weight_fingerprint.py
|   +– activation_capture.py
|   +– functional_similarity.py
|   +– equivalence_detector.py
|   +– information_overlap.py
|   +– subgraph_comparator.py
|   +– algorithmic_equivalence.py
|
+– transforms/
|   +– factorization.py
|   +– shared_basis.py
|   +– tensor_decomposition.py
|   +– block_decomposition.py
|   +– subgraph_fusion.py
|   +– algebraic_rewrite.py
|   +– reconstruction.py
|
+– profiler/
|   +– flops.py
|   +– memory.py
|   +– transfer.py
|   +– latency.py
|   +– cache.py
|   +– hardware_backend.py
|
+– validation/
|   +– numerical.py
|   +– perplexity.py
|   +– regression.py
|   +– domain_coverage.py
|   +– robustness.py
|
+– reports/
|   +– similarity_report.py
|   +– equivalence_report.py
|   +– compression_report.py
|   +– hardware_report.py
|
+– fid.py

================================================================
5. LES QUATRE MOTEURS D’ANALYSE

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
* Erreur par categorie d’entree.
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

Il faut mesurer l’erreur numerique et fonctionnelle.

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
* Temps SSD -> RAM.
* Temps RAM -> GPU.
* Temps RAM -> HTP.
* Temps de reconstruction.
* Temps d’attente du calculateur.
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

ETAPE 1 — PROFILAGE INITIAL

* Charger le modele.
* Extraire les tenseurs.
* Construire le graphe.
* Capturer les activations.
* Produire les empreintes.

Aucun poids modifie.

ETAPE 2 — DETECTION

* Comparer les couches.
* Comparer les experts.
* Comparer les sous-graphes.
* Identifier les facteurs communs.
* Identifier les fonctions proches.
* Identifier les chemins algorithmiques equivalents.

ETAPE 3 — GENERATION

Generer des representations candidates :

* Low rank.
* Shared basis.
* Tensor factorization.
* Block decomposition.
* Algebraic rewrite.
* Subgraph fusion.
* Expert merging.

ETAPE 4 — EVALUATION

Pour chaque candidat :

* Erreur numerique.
* Erreur fonctionnelle.
* Perplexite.
* Parametres.
* RAM/VRAM/HTP.
* FLOPs.
* Latence.
* Transferts.

ETAPE 5 — RECONSTRUCTION

* Reconstruire le sous-reseau.
* Recalculer les dependances.
* Verifier les dimensions.
* Verifier le routage.
* Verifier les residuals et normalisations.
* Tester les cas limites.

ETAPE 6 — VALIDATION

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

1. Choisir deux experts d’un modele MoE.
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

PRIORITE 1 — DECOUVERTE DE REDONDANCE

* TD-MoE.
* FCM.
* PuzzleMoE.
* Sub-MoE.
* LoRP.

PRIORITE 2 — REPRESENTATION PARTAGEE

* MoBE.
* MixT.
* Global Rank/Sparsity.
* RED.

PRIORITE 3 — RECONSTRUCTION ET REDUCTION

* XMerge.
* SWM.
* MoNE.

PRIORITE 4 — OPTIMISATION DU RUNTIME

* FloE.
* FIRM-MoE.
* SMoE.

FID doit etre le moteur commun d’analyse qui alimente ces
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

DEMONTRÉ :
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
8. Comment valider qu’il conserve ses capacites.

VISION :

NE PLUS COMPRESSER UNIQUEMENT LES POIDS.

COMPRENDRE LES CALCULS ET LES INFORMATIONS QU’ILS
REPRESENTENT, PUIS RECONSTRUIRE UNE ARCHITECTURE
QUI LES CALCULE AVEC LE MOINS DE RESSOURCES POSSIBLE.


Similarité purement statistique vs fonctionnelle réelle
Les empreintes actuelles (distribution, sparsity, cosine) capturent des redondances numériques/structurelles, mais pas forcément l'équivalence algorithmique ou fonctionnelle (même transformation input>output sur des données réelles). Deux modules peuvent avoir des poids très différents mais calculer quasi la même chose (ou l'inverse).

2. Absence d'analyse dynamique / activation-based
Tout repose sur les poids statiques. Or la littérature montre que la redondance réelle apparaît souvent au niveau des activations, du routing MoE, ou des chemins de calcul (

Spécificité MoE sous-exploité
Le projet mentionne MoE, mais le code de base ne semble pas encore traiter le routing, le co-activation des experts, les experts "morts", ni le partage cross-layer.
4. Pas de métrique de "coût réel" ni de compression actionnable
Détecter une redondance est une chose ; quantifier le gain réel (FLOPs, mémoire, latence) et proposer une fusion/pruning/sharing safe en est une autre.
5. Scalabilité et validation limitée
Tests sur tiny-gpt2 / CPU. Pas encore de benchmarks sur de vrais MoE (Mixtral, DeepSeek-MoE, etc.) ni de mesures de downstream impact après fusion.
6. Manque de distances
Cosine et relative error collapsent en haute dimension. Les méthodes modernes utilisent Gromov-Wasserstein, CKA, SVD/spectral fingerprints, ou équivalence différentielle.
7. Pas encore de graphe de calcul / circuit discovery
L'extracteur de graphe est présent, mais l'analyse reste au niveau module/ tenseur plutôt qu'au niveau des chemins de calcul interchangeables.
Croisement avec la littérature

Redondance MoE & partage d'experts
• Meta Expert : Cross-Layer Expert
Sharing → https://arxiv.org/abs/.
2610.27598 (les experts inter-couches sont souvent redondants ; partager des
"Meta Experts" permet d'atteindre les perfs d'un MoE vanilla avec seulement 70-80 % des experts).
• Neuronal Redundancy Reduction for
PEFT-MoE → https://aclanthology.org/
2026.emnlp-main.2027l.
• Exploiting MoE Redundancy for
Multimodal →
https://arxiv.org/absl
2503.22517
• R'MoE (Redundancy-Removal MoE) →
https://arxiv.org/abs/2507.13107
https://arxiv.org/abs/2507.13107
Similarité fonctionnelle & sous-réseaux
• Identifying Sub-networks via
Functionally Similar Representations (Gromov-Wasserstein) → https:/l.
arxiv.org/abs/2410.16484
• Survey: Similarity of Neural Network
Models (représentationnelle + fonctionnelle) → https://.
ar5iv.labs.arxiv.org/html/2305.06329
• Neural Network Compression by
Approximate Differential Equivalence →
https://arxiv.org/abs/2606.01402
• Layer Redundancy from a functional perspective → https://arxiv.org/abs/
2604.24938
Fingerprinting / empreintes de poids robustes
AWM : Accurate Weight-Matrix
Fingerprint (LAP + CKA) → https://.
arxiv.org/abs/2510.06738
• Spectral Fingerprinting (SVD des GATE weights) → https://arxiv.org/html/.
2610.30626v1
• SELF (SVD + Eigenvalue invariants) →
https://arxiv.org/html/2512.03620
Pruning / compression par redondance fonctionnelle
• Cluster-Based Structural Redundancy
Identification → https:/l.
pmc.ncbi.nlm.nih.gov/articles/
PMC98576171.
• RED (data-free structured compression via redundancies) → https://arxiv.org/. html/2105.14797v1
• Low functional redundancy network slimming → https:/l.

https://www.sciencedirect.com/science/article/pii/S1110016824017162









FIN DU CAHIER DES CHARGES FID v0.1

