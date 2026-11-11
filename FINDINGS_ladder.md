# FINDINGS — Le paramétrage minimal de l'interprétation : intrinsèque ou relationnel ?

*2026-06-10. Résultats honnêtes, verdict pré-enregistré, une prédiction réfutée et un recadrage forcés par la vérification adversariale (6 sceptiques indépendants).*
*Voir `PREREGISTRATION_ladder.md` (déclaré avant les runs), `results_ladder.json`, `viz/ladder_interpretation.html`.*

---

## Verdict en une phrase

**Le paramétrage minimal pour qu'un système interprète des données n'existe pas dans l'absolu : il est RELATIONNEL.** L'interprète minimal suffisant rôde entre L1 et L3 selon la classe de flux, et il existe des inversions robustes où un biais inductif *plus simple* bat un apprenant *plus riche*. Mais la relationalité a **deux régimes**, et un seul est irréductible.

## Ce qui était testé

« Quel est le degré minimal de paramétrage nécessaire pour interpréter des données ? » rendu falsifiable : une échelle d'interprètes à structure innée croissante (L0 marginal → L1 mémoire → L2 comparaison → L3 règle globale → L4 non-linéarité → L5 capacité) × 10 classes de flux à structure hétérogène, mesuré par `interp` = R² un-pas held-out. Question décisive : l'ordre des interprètes est-il **total** (mainstream : une échelle universelle, le minimum se lit sur la complexité de la donnée) ou admet-il des **inversions** (bord : aucune échelle indépendante du substrat — No Free Lunch, ρ(X,M)) ?

## Résultat principal — RELATIONNEL (pré-enregistré, H1∧H3∧H6)

| | verdict |
|---|---|
| H1 plancher (`iid` muet) | ✅ aucun barreau n'interprète le bruit |
| H2 minimum suffisant glisse | ✅ rôde **L1↔L3** selon la classe |
| **H3 inversion robuste** | ✅ **14 inversions** (Δ 0.06–0.98, toutes > 2 SEM) — *décisif* |
| H4 crossover (non-dominance) | ✅ 7 paires de barreaux mutuellement non-dominantes |
| H5 onset Ξ glisse | ❌ l'intégration s'allume ~L2 partout (voir réfutation) |
| H6 surrogate | ✅ 0 inversion survit au mélange temporel ; 14/14 tiennent sur la structure temporelle (real − surrogate) |

Le minimum suffisant (barreau le plus bas atteignant 0.05 du plafond) atterrit sur **trois barreaux distincts**, chacun par marge robuste :
- **L1 (persistance)** — `randomwalk`, `ar1` : le biais d'inertie suffit.
- **L2 (kNN)** — `logistic`, `xor3`, `rbn`, `periodic` : il faut comparer des contextes.
- **L3 (linéaire)** — `noisy_ar2`, `ar3_osc`, `tar` : il faut une règle globale ; `noisy_ar2` bat la persistance (Δ 0.31) ET le kNN (Δ 0.07, ~12σ), un optimum au *milieu* de l'échelle.

## Le vrai apport — DEUX régimes de relationalité

Toutes les inversions ne se valent pas. La vérification budget (`robustness_ladder.py --arm budget`, jusqu'à hid=128/budget=60000, ~40×) les sépare nettement :

### 1. Capacité-relationnelles (défaisables par le compute)
`logistic` et `xor3` : kNN(L2) bat le MLP à budget fixe (Δ 0.27–0.51), mais le MLP **rattrape** à hid=64/budget=20000 (0.99). C'est le régime du *bitter lesson* : à compute suffisant, l'apprenant général gagne. Réel mais défaisable.

### 2. Biais-fondamentales (irréductibles — AUCUN budget ne corrige)
`randomwalk` : **linear(L3) et persistence(L1) battent le MLP à *tout* budget** — le MLP à 40× capacité score **0.000**. Mécanisme : le processus intégré est non-stationnaire (37 % des cibles test hors de l'enveloppe d'entraînement), donc tout modèle qui apprend une fonction globale échoue à extrapoler, tandis que la persistance (zéro paramètre) est l'estimateur optimal par construction (la moyenne conditionnelle d'une marche aléatoire EST la dernière valeur). C'est la signature No Free Lunch / ρ(X,M) au niveau le plus nu.

> **L'asymétrie structurelle qui explique tout** : la classe de fonctions est emboîtée vers le haut (linéaire ⊂ MLP). Un barreau plus riche peut donc toujours *égaler* un plus simple à données suffisantes — d'où l'absence d'inversion biais-fondamentale battant un barreau *au-dessus* par sur-ajustement seul. Les inversions irréductibles n'apparaissent **que là où la donnée viole une hypothèse dont le modèle riche a besoin** (stationnarité, support borné). Le minimum suffisant rôde donc *vers le bas* quand le monde est plus structuré que le modèle ne le suppose — pas l'inverse.

## Vérification adversariale — ce qui a tenu, ce qui est tombé

Six sceptiques indépendants, chacun chargé de *réfuter* une affirmation (workflow `verify-ladder-inversions`).

- **Fuite train/test** → *tient (sévérité nulle).* kNN=1.0 sur logistic/xor3 est de la mémorisation légitime d'une carte déterministe : invariant à gap=20/50/100, survit sur trajectoire entièrement disjointe, insensible au contrôle de fuite délibérée. Pas de canal de fuite.
- **Artefact numérique (randomwalk)** → *tient (nulle).* persistence > kNN est **scale-invariant** (écart fixé à 0.745 sur du bruit de 0.01 à 100), tient sur RMSE brut (ratio 6.8×), sur z-score, sur train long, à tout k=1..50. Note d'honnêteté : le R² *clippé* [0,1] sous-estime l'échec du kNN (R² non clippé −0.03).
- **Ordre de l'échelle (kNN « plus simple » ?)** → *partiel (mineur).* Objection réelle : le kNN stocke tout l'entraînement → sur l'axe capacité/VC il est le *plus* complexe, pas le plus simple. **Recadrage adopté** : ne pas faire reposer la thèse sur le rang du kNN. La tête de gondole est désormais l'inversion **capacité-non-ambiguë** linear(L3) > MLP sur randomwalk (linéaire strictement inclus dans la classe MLP, pourtant MLP empire à 40× budget). La relationalité ne faiblit pas — les wins kNN logistic/xor3 sont reclassés *secondaires, capacité-relationnels*.
- **Ξ_data prédit la direction de l'optimum** → **RÉFUTÉ (majeur).** *Retiré.* Seules 4/9 classes collent ; test de permutation **p = 0.78** (pire que le hasard) ; les classes redondantes (Ξ<0) sont majoritairement gagnées par un *intégrateur* (kNN). Seul `xor3` est génuinement synergique (Ξ=+0.64). Survit uniquement la version étroite : *les processus intégrés extrêmement redondants (randomwalk, Ξ=−2.23) favorisent le biais inertiel.* Les extrêmes parlent, le signe ne prédit pas.
- **Robustesse à la fenêtre D** → *partiel (mineur).* persistence>knn (randomwalk) et knn>linear (logistic) tiennent à D∈{2,3,4,6}. Mais **xor3 est fragile** : son inversion ne vit qu'en D∈{3,4,5} ; à D≥6 le linéaire résout exactement le lookup à 7 états (xor3 = orbite de période 7). Flag posé.
- **Complétude** → *partiel (mineur).* (a) Le triplet (Γ,Ξ,β) est **décoratif** pour le verdict : seul `interp` pilote H1/H3/H6 ; H5 (Ξ-onset) a échoué → le triplet ne porte pas la conclusion. Démoté à descriptif. (b) Plusieurs argmax sont des **égalités de bruit** (periodic/ar1/ar3_osc/tar) → le « skyline » trace désormais le *minimum suffisant* (robuste), pas l'argmax (surévalué). (c) Ajout forcé d'une classe à optimum *au milieu* (`noisy_ar2`) pour prouver que le minimum rôde dans les deux sens.

## Lien à la thèse et au corpus

Réponse à la question d'origine (« inné vs acquis dans les mécanismes intelligents »), au niveau le plus nu :
- **Kant/Solomonoff confirmés négativement** : `iid` muet pour tous → zéro biais, zéro interprétation (NFL).
- **Le minimum n'est pas un nombre, c'est une fonction du couplage.** L'interprète minimal suffisant rôde L1↔L3 ; il n'existe pas d'échelle universelle des biais inductifs. C'est ρ(X,M) (**[[pont_maitre_finding]]**) instancié sur le substrat le plus dépouillé — 4ᵉ occurrence de « l'intrinsèque ne prédit pas le relationnel », mais raffinée : la relationalité se **scinde** en une part défaisable par le compute (capacité) et une part irréductible (biais). L'irréductible est exactement là où le prior du substrat doit épouser la structure du monde — l'inné load-bearing.
- Pattern « 1 ingrédient load-bearing » récurrent (dynamoscope, meta-fractal) : des 14 inversions, **4 sont biais-fondamentales** (persistence/linear > MLP sur randomwalk) ; le reste est capacité-relationnel ou contestable.

## Limites honnêtes

- 10 classes, 6 barreaux, 5 seeds (12 pour les inversions cœur), petits modèles. Le triplet (Γ,Ξ,β) est descriptif, pas décisif.
- Pas d'inversion biais-fondamentale battant un barreau *au-dessus* (impossible vu l'emboîtement des classes) : la relationalité irréductible est **mono-directionnelle** (le minimum rôde vers le bas).
- `xor3` est une orbite de période 7 (carte finie minuscule), pas une classe riche — son inversion est D-bandée. `rbn` porte une composante *statique* (nœuds gelés) qui trompe le surrogate naïf — même signature que l'exception RBN de l'essai bord-du-chaos (tables fixes → structure persistante).
- Un seul flux génuinement synergique (`xor3`) : impossible de tester sérieusement le rôle de la synergie sans plusieurs.

## Question ouverte

L'asymétrie est-elle fondamentale ou un artefact de l'emboîtement des classes de fonctions ? Si l'on bâtissait une échelle d'interprètes **non emboîtée** (des biais inductifs orthogonaux, pas des capacités croissantes — p. ex. un banc de modèles à *symétries* différentes plutôt qu'à *flexibilité* croissante), verrait-on enfin des inversions biais-fondamentales *bidirectionnelles* — un optimum qui rôde vraiment dans toutes les directions, sans qu'aucun barreau ne puisse jamais en égaler un autre ? Ce serait le No Free Lunch pleinement matérialisé : non plus « le simple est sous-estimé », mais « aucun biais n'en domine un autre, jamais ». C'est la prochaine marche — porter l'échelle sur des architectures à symétries distinctes (table périodique des biais, **[[architectural_classes_program]]**).
