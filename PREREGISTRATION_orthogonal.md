# Pré-enregistrement — No Free Lunch plein : des inversions biais-fondamentales BIDIRECTIONNELLES ?

*Déclaré AVANT d'exécuter. Aucun seuil modifié après les résultats. 2026-06-10.*
*Suite de `PREREGISTRATION_ladder.md` : l'échelle emboîtée (linéaire ⊂ MLP) n'a produit que des inversions VERS LE BAS (un barreau riche peut toujours égaler un simple à compute suffisant). Ici on casse l'emboîtement.*

---

## Question décisive

Si les interprètes ne sont plus ordonnés par **capacité croissante** (emboîtés) mais par **biais inductif orthogonal** (engagements structurels distincts, aucun inclus dans un autre), obtient-on des inversions **bidirectionnelles** — A bat B sur une classe ET B bat A sur une autre, *aucun budget ne renversant ni l'une ni l'autre* ?

- **Échelle emboîtée (expérience précédente)** → inversions mono-directionnelles : le minimum rôde vers le bas, le riche rattrape toujours.
- **Panel orthogonal (ici)** → si bidirectionnel : **No Free Lunch plein** — aucun biais ne domine, jamais ; l'optimum est *purement* une fonction du couplage (interprète × classe). C'est la forme forte de ρ(X,M).

## Panel d'interprètes à biais ORTHOGONAUX (aucun emboîté)

| nom | prior structurel | « vit dans » | échec fondamental |
|-----|------------------|--------------|-------------------|
| `inertial` | marche aléatoire : x̂ = dernier + dérive (0 param) | l'état récent | ne peut PAS osciller |
| `affine` | dynamique linéaire globale (AR moindres carrés) | l'état récent | ne peut PAS faire le non-linéaire/XOR |
| `fourier` | harmonique global : somme de K sinusoïdes (fonction du TEMPS) | la fréquence | ne peut PAS suivre un processus sans fréquence stable (marche) |
| `local` | localité : successeur moyen des k contextes voisins | l'état récent | ne peut PAS extrapoler hors du support échantillonné |
| `fsm` | machine à états discrète : table de vérité sur le motif binarisé | le motif discret | ne peut PAS lire l'amplitude continue |

Clé : ces priors ne s'emboîtent pas. Le `fourier` modélise le temps, pas l'état ; l'`inertial` n'a aucune fréquence ; le `fsm` jette l'amplitude. **Aucun n'est un cas particulier d'un autre** → aucun ne peut, par montée en capacité, faire le travail d'un autre.

## Classes de flux (chacune a un « foyer » naturel)

Réutilise `generators.py` (10 classes). Foyers attendus : `periodic`→fourier · `randomwalk`→inertial · `noisy_ar2`/`ar3_osc`→affine · `logistic`→local · `xor3`/`rbn`→fsm.

## Mesure & budget

- `interp` = R² un-pas held-out (split temporel 70/30, anti-fuite), comme l'expérience sœur.
- **Bras budget** (`--arm budget`) : chaque interprète poussé à sa capacité max (fourier K↑, local k/train↑, affine D↑, fsm ordre↑). Un échec qui PERSISTE au budget max = biais-fondamental (pas un sous-entraînement).

## Hypothèses et seuils (pré-enregistrés)

| # | Énoncé | KEEP | Kill |
|---|--------|------|------|
| **H1** plancher | `iid` muet pour tous | max interp(iid,·) < 0.05 | sinon fuite |
| **H2** pas d'universel | aucun interprète ne gagne (à 0.05 près) sur TOUTES les classes structurées | ∀ interprète ∃ classe où il est battu de > 0.1 | un interprète domine partout → pas relationnel |
| **H3** inversion BIDIRECTIONNELLE biais-fondamentale (décisif) | ∃ A,B,c1,c2 : interp[A,c1] > interp[B,c1]+0.1 ET interp[B,c2] > interp[A,c2]+0.1, **les deux écarts survivant au bras budget des deux interprètes** | ≥1 paire bidirectionnelle budget-robuste | aucune → l'orthogonalité n'a pas suffi (les inversions restent mono-dir comme l'échelle emboîtée) |
| **H4** intransitivité | la relation de domination admet un CYCLE (≥3 interprètes, A>B>C>A sur leurs classes croisées) ou ≥2 paires mutuellement non-dominantes | cycle ou ≥2 crossovers | — |
| **H5** surrogate | toute capture disparaît sur le flux mélangé | 0 inversion sur surrogate | sinon artefact |

## Verdict global (décidé à l'avance)

- **NFL PLEIN** : H1 ∧ H3 (bidirectionnelle budget-robuste) ∧ H5. → aucun biais ne domine ; l'optimum est purement relationnel ; la version forte de « l'intrinsèque ne prédit pas le relationnel ». Distinct de l'échelle emboîtée (mono-dir).
- **MONO-DIRECTIONNEL** : H1 vrai, inversions présentes mais aucune bidirectionnelle budget-robuste. → l'orthogonalité n'a pas changé le régime ; à creuser.
- **UN BIAIS DOMINE** : H2 faux (un interprète gagne partout). → contre toute attente, un prior universel existe sur ce banc de classes.

Prédiction (non contraignante) : **NFL plein**, paire-phare `fourier`↔`inertial` (fourier gagne `periodic`, inertial gagne `randomwalk`, aucun ne renverse l'autre à budget max), et cycle probable fourier > (sur periodic) ... local > (sur logistic) ... inertial > (sur randomwalk) ... La structure devrait être antisymétrique, pas un ordre.
