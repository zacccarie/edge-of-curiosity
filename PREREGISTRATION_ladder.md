# Pré-enregistrement — Le paramétrage minimal de l'interprétation est-il intrinsèque ou relationnel ?

*Déclaré AVANT d'exécuter l'expérience. Aucun seuil ne sera modifié après avoir vu les résultats.*
*Date : 2026-06-10. Programme : edge-of-curiosity (sœur de l'essai « bord du sens » ; descendante directe du Pont Maître).*

---

## Question décisive

« Quel est le degré minimal de paramétrage nécessaire pour qu'un système puisse **interpréter** des données ? » — et surtout : **ce minimum est-il une propriété intrinsèque du système, ou une propriété relationnelle du couplage (système, classe de données) ?**

Reformulation falsifiable. On construit une *échelle d'interprètes* à structure innée croissante (un primitif ajouté par barreau) et une *batterie de classes de flux* à structure hétérogène. On mesure, pour chaque couple (classe × barreau), à quel point l'interprète capte la structure du flux, et la signature (Γ, Ξ, β) de cette capture.

- **H_intrinsèque (mainstream).** L'échelle est un **ordre total** : tout barreau supérieur domine tout barreau inférieur sur *toute* classe. Alors « le niveau minimal suffisant » est une simple lecture de la complexité (intrinsèque) de la donnée ; l'échelle des biais inductifs est universelle. Le seuil d'apparition de l'interprétation est le **même barreau** quelle que soit la classe.
- **H_relationnelle (bord).** L'ordre **n'est pas total** : il existe une **inversion** — un interprète *plus simple* bat strictement un interprète *plus complexe* sur au moins une classe, parce que son biais inductif épouse mieux cette classe. Alors « le paramétrage minimal pour interpréter » n'est définissable que **relativement à la classe** ; aucune échelle indépendante du substrat. C'est le No Free Lunch rendu concret, et ρ(X,M) au niveau le plus nu.

## Dispositif

### Échelle d'interprètes (structure innée croissante — un primitif par barreau)
| L | nom | primitif ajouté | biais inductif |
|---|-----|-----------------|----------------|
| L0 | `marginal` | — (mémoire nulle) | la donnée n'a pas de structure temporelle (plancher) |
| L1 | `persistence` | **mémoire** (dernier état) | le monde est persistant / inertiel |
| L2 | `knn` | **comparaison** (métrique sur les contextes) | contextes similaires → futurs similaires (local, non-paramétrique) |
| L3 | `linear` | **règle paramétrique globale** (moindres carrés) | le monde est globalement affine |
| L4 | `mlp_small` | **non-linéarité bornée** (MLP hid=8) | fonction lisse non-linéaire du passé récent |
| L5 | `mlp_large` | **capacité** (MLP hid=32) | idem, ordre supérieur / compositionnel |

### Classes de flux (structure hétérogène — l'axe de variation)
`iid` (bruit, plancher) · `periodic` · `ar1` (linéaire court) · `randomwalk` (intégré) · `ar3_osc` (linéaire long) · `logistic` (non-linéaire 1D déterministe) · `tar` (affine par morceaux) · `xor3` (compositionnel binaire, `x_t = x_{t-1} ⊕ x_{t-3}`) · `rbn` (booléen en réseau, K=2).

### Mesure
- **interp(classe, barreau, seed)** = `clip(1 − NMSE_un-pas_test, 0, 1)` sur un split temporel held-out (fit 70 % / test 30 %, aucune fuite). 0 = pas mieux que prédire la marginale ; 1 = structure entièrement captée. Définit le seuil d'apparition L\*.
- **Triplet (Γ, Ξ, β)** (instanciation opérationnelle ; Ξ reste un *template*, cf. xi-observable) :
  - **β** = pente spectrale (log-log, bande médiane) du **résidu** un-pas. Interprétation parfaite ⇒ résidu blanchi (β→0). Le signe/caractère de β dit *quelle* structure reste sur la table.
  - **Ξ** = information prédictive **synergique** captée = `I(futur ; contexte joint) − Σ_k I(futur ; lag_k)` (MI par histogrammes). Marque l'apparition de l'**intégration** (irréductible aux lookups mono-feature) — le moment où « interpréter » diffère de « consulter ».
  - **Γ** = **organisation** du rollout libre = `max(0, 1 − distance_ACF(rollout, vérité))`. Un modèle qui s'effondre (point fixe) ou diverge a Γ bas ; un modèle qui reproduit l'attracteur a Γ haut. (Stabilité inter-seed reportée à part, en diagnostic.)

## Hypothèses et seuils (pré-enregistrés)

| # | Énoncé | Critère KEEP | Kill-switch |
|---|--------|--------------|-------------|
| **H1** plancher | `iid` n'est interprété par aucun barreau | max_L interp(iid, L) < marge (0.05) | si un barreau « interprète » le bruit → fuite/artefact → run mort |
| **H2** glissement | le seuil d'apparition L\*(classe) **varie** selon la classe | étendue de L\* ≥ 2 barreaux sur les classes structurées | L\* constant → minimum universel (intrinsèque) |
| **H3** inversion (décisif) | ∃ inversion robuste : un barreau **plus bas** bat un **plus haut** sur une classe | interp[a,c] > interp[b,c] + marge, a<b, écart > 2·SEM ET > 0.05 | aucune inversion → ordre total → **intrinsèque** |
| **H4** non-dominance | l'ordre des barreaux n'est total sur aucune paire couvrant toutes les classes | ≥1 paire (a,b) mutuellement non-dominante (crossover) | — |
| **H5** intégration | Ξ s'allume à un barreau qui **glisse** selon la classe | onset(Ξ>0) varie ≥ 2 barreaux | onset Ξ constant → intégration à seuil universel |
| **H6** surrogate | toute structure captée disparaît sur le flux mélangé | interp_surrogate < marge ∀ (classe, barreau) ; **0 inversion sur surrogate** | inversion reproduite sur surrogate → artefact → H3 tué |

- **marge** = 0.05 (absolu) ; SEM = écart-type inter-seed / √(n_seed).
- **Surrogate** = permutation temporelle du flux (détruit la dynamique, garde les marginales). Plancher anti-artefact, hérité de l'essai bord-du-sens (où la NMSE non normalisée avait fabriqué un faux skill — corrigé ici par normalisation par la variance du signal).

## Verdict global (décidé à l'avance)

- **RELATIONNEL (bord) confirmé** : H1 ∧ H3 (≥1 inversion robuste) ∧ H6 (inversion absente du surrogate). → le paramétrage minimal de l'interprétation n'existe pas dans l'absolu ; il est défini par le couplage. 4ᵉ occurrence de « l'intrinsèque ne prédit pas le relationnel ».
- **INTRINSÈQUE (mainstream)** : H1 vrai, **aucune** inversion (ordre total), L\* monotone d'un scalaire de complexité. → il existe une échelle universelle ; le minimum est une lecture de la complexité de la donnée.
- **MIXTE** : glissement (H2) sans inversion (H3) → le minimum *glisse* mais reste une fonction bien définie de la classe (relationnel faible) ; on cartographie la frontière.

Prédiction personnelle (non contraignante) : **relationnel**, avec inversion `persistence > linear, mlp` sur `randomwalk` (le biais d'inertie bat la règle globale sur un processus intégré) et `knn/mlp > linear` sur `logistic`/`xor3`. Mais le dispositif peut me falsifier — c'est l'enjeu.
