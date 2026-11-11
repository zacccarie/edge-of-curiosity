# FINDINGS — No Free Lunch plein : inversions biais-fondamentales bidirectionnelles

*2026-06-10. Suite directe de `FINDINGS_ladder.md`. Pré-enregistré dans `PREREGISTRATION_orthogonal.md`. Données : `results_orthogonal.json`, viz : `viz/orthogonal_nfl.html`.*

---

## Verdict en une phrase

**En cassant l'emboîtement des interprètes (biais orthogonaux au lieu de capacités croissantes), la relationalité devient irréductible : les inversions sont bidirectionnelles, la domination est intransitive, aucun biais ne domine — jamais.** C'est le No Free Lunch pleinement matérialisé.

## Pourquoi cette expérience

L'échelle emboîtée (linéaire ⊂ MLP) ne pouvait produire que des inversions *vers le bas* : un barreau riche peut toujours *égaler* un simple à compute suffisant. Donc une part de la relationalité y restait **défaisable par le compute** (capacité). Pour tester la part *irréductible*, on remplace l'échelle de capacité par un **panel de biais orthogonaux** — cinq engagements structurels dont aucun n'est un cas particulier d'un autre :

| interprète | prior | vit dans | échec fondamental |
|---|---|---|---|
| `inertial` | marche aléatoire (0 param) | l'état récent | ne peut pas osciller |
| `affine` | linéaire global | l'état récent | ne peut pas le XOR |
| `fourier` | harmonique global | la fréquence/le temps | ne peut pas suivre un processus sans fréquence stable |
| `local` | localité (kNN) | l'état récent | ne peut pas extrapoler hors support |
| `fsm` | motif discret (table) | le motif binarisé | ne peut pas lire l'amplitude |

## Résultats (10 classes × 5 biais × 5 seeds, R² un-pas held-out)

| H | énoncé | verdict |
|---|---|---|
| H1 | plancher `iid` muet | ✅ |
| H2 | aucun interprète universel | ✅ (chaque biais perd de >0.1 sur ≥1 classe) |
| **H3** | **inversion bidirectionnelle budget-robuste** | ✅ **7 paires, toutes budget-robustes** |
| H4 | intransitivité (cycle de domination) | ✅ **11 cycles** |
| H5 | surrogate propre | ✅ 0 inversion bidirectionnelle survit au mélange |

Chaque classe a un **foyer distinct** : `periodic`→fourier/local, `randomwalk`→inertial, `noisy_ar2`/`ar3_osc`/`tar`→affine, `logistic`→local, `xor3`→fourier, `rbn`→affine/local. Aucune ligne ne domine la matrice.

### La paire-phare — `fourier` ⇄ `inertial`
- `fourier` gagne `periodic` (0.99 vs 0.86) — `inertial`, à 0 paramètre, ne peut PAS osciller. Aucun budget ne le corrige (il n'a pas de bouton).
- `inertial` gagne `randomwalk` (0.98 vs 0.00) — `fourier`, même à K=24 harmoniques, ne peut PAS suivre un processus sans fréquence stable (0.00 à tout budget).
- **Bidirectionnelle, budget-robuste des deux côtés.** Ni l'un ni l'autre ne fait jamais le travail de l'autre.

### Le cœur intransitif (rock-paper-scissors)
`inertial → fourier → affine → inertial` :
- `inertial` bat `fourier` sur `randomwalk` (0.98 vs 0.00)
- `fourier` bat `affine` sur `xor3` (1.00 vs 0.00)
- `affine` bat `inertial` sur `noisy_ar2` (0.87 vs 0.56)

Trois interprètes, aucun « meilleur » : la relation de domination a un **cycle**, pas un ordre. 11 tels cycles au total.

## La différence décisive avec l'échelle emboîtée

| | échelle emboîtée (capacité) | panel orthogonal (biais) |
|---|---|---|
| inversions | mono-directionnelles (vers le bas) | **bidirectionnelles** |
| structure de domination | quasi-ordre (le riche rattrape) | **cyclique, intransitive** |
| compute | dissout les inversions de capacité | **n'en dissout aucune** |
| relationalité | en partie défaisable | **irréductible** |

L'emboîtement était l'échappatoire qui laissait croire à un « apprenant universel à compute suffisant ». Levé l'emboîtement, l'échappatoire disparaît : **il n'existe pas de meilleur interprète, même approximativement** — l'optimum est purement le couplage (biais × classe).

## Lien à la question d'origine et au corpus

Réponse forte à « inné vs acquis » : le paramétrage minimal/optimal n'est pas une *quantité* sur une échelle, c'est une **direction dans l'espace des biais**. Deux systèmes peuvent être chacun optimal et chacun inutile, sur des mondes différents, sans qu'aucun ne soit convertible en l'autre. L'inné qui compte, c'est **quelle symétrie le système engage** — défini seulement relativement à la structure du monde.

C'est la forme la plus forte de ρ(X,M) ([[pont_maitre_finding]]) et la clôture de l'invariant « l'intrinsèque ne prédit pas le relationnel » : non seulement l'intrinsèque ne le prédit pas, mais il n'existe aucun intrinsèque privilégié. Pont vers la table périodique des biais inductifs ([[architectural_classes_program]]) : ces cinq priors sont cinq classes architecturales, et le cycle de domination est leur structure de couplage.

## Limites honnêtes

- `affine` et `local` sont des **généralistes forts** (dominent `inertial`/`fourier` respectivement, arêtes grises sur le graphe) — la structure n'est pas un tournoi complet, mais elle reste sans universel et avec cycles. Le NFL ici est *empirique sur ce banc*, pas un théorème (qui, lui, est déjà connu et que ceci instancie).
- `fourier` a nécessité un raffinement de fréquence (FFT brute trop grossière → dérive de phase sur l'horizon test) — corrigé, mais c'est une faiblesse d'implémentation, pas du prior ; documenté dans `orthogonal_interpreters.py`.
- Le surrogate plafonne à 0.20 sur `rbn` (structure statique des nœuds gelés) — même caveat que l'échelle ; 0 inversion bidirectionnelle survit, donc sans effet sur la conclusion.
- 5 biais, 10 classes : un échantillon, pas l'espace complet. Ajouter des priors (ondelettes, symétrie de groupe explicite) ne ferait qu'épaissir le cycle.

## Question ouverte

Le graphe de domination est-il toujours cyclique dès que les biais sont orthogonaux, ou existe-t-il des bancs où un méta-biais (un mélange adaptatif, type *gating* / mixture-of-experts) échappe au cycle en *sélectionnant* le bon prior par classe ? Si oui, ce méta-biais serait le vrai « inné universel » — non pas un prédicteur universel, mais un *routeur* universel. C'est exactement l'architecture que cherche `adaptive-nn` : non pas le meilleur biais, mais la capacité de commuter de biais selon le monde rencontré. Le NFL ne tue pas l'universalité — il la déplace du prédicteur vers le **sélecteur**.
