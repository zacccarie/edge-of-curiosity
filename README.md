# edge-of-curiosity

**La curiosité cherche-t-elle le bord du chaos ?**
L'expérience décisive des essais *Au bord du sens* / *Rester au bord du sens* — adaptive-nn Phase 2, le chaînon manquant entre une mesure passive du bord du chaos et un agent qui le *vise* activement.

## Thèse testée

Conjecture maîtresse : *argmax(progrès d'apprentissage) coïncide avec le pic de susceptibilité critique* (le bord du chaos). Si oui → la thèse « l'intelligence est un équilibre activement tenu au bord d'une instabilité » reçoit une assise. Si non → le bord est une corniche subie, pas un but.

## Dispositif

- **3 substrats** (règle des ≥3 régimes) — `substrates.py` : carte logistique (`r`), réseau de cartes couplées (`ε`), réseau booléen aléatoire (`K`). Bord du chaos analytiquement connu pour chacun.
- **Prédicteur à capacité bornée** — `predictor.py` : skill = rollout multi-pas ; LP intrinsèque = amélioration *tardive* du skill (annule ordre et chaos).
- **3 agents** — `curiosity.py` : `lp` (curiosité), `error` (nouveauté brute = piège « télé qui neige »), `random` (contrôle).
- **Verdicts pré-enregistrés** — `PREREGISTRATION.md` + `experiment.py`.

## Lancer

```bash
python experiment.py --smoke   # validation rapide (~30s)
python experiment.py           # run complet, 3 substrats × 5 seeds -> results.json
```

numpy uniquement. Pré-enregistrement à lire **avant** les résultats.

---

## Expérience sœur — Le paramétrage minimal de l'interprétation

*Intrinsèque ou relationnel ? Le degré minimal de structure innée pour interpréter des données est-il une propriété du système, ou du couplage (système, classe de données) ?*

- **Échelle d'interprètes** (`interpreters.py`) : L0 marginal → L1 mémoire → L2 comparaison → L3 règle globale → L4 non-linéarité → L5 capacité (un primitif inné par barreau).
- **10 classes de flux** (`generators.py`) : bruit / périodique / linéaire / intégré / oscillant / non-linéaire / compositionnel / booléen.
- **Test décisif** : l'ordre des interprètes est-il total (intrinsèque) ou admet-il des **inversions** (relationnel — un biais plus simple bat un apprenant plus riche) ?

```bash
python ladder_experiment.py --smoke      # ~3s
python ladder_experiment.py              # 10 classes × 6 barreaux × 5 seeds -> results_ladder.json
python robustness_ladder.py --arm budget # capacité vs biais : quelles inversions survivent au compute ?
```

Verdict : **RELATIONNEL**. Le minimum suffisant rôde L1↔L3 ; 14 inversions robustes, dont 4 biais-fondamentales (`randomwalk` : linear/persistence battent le MLP à 40× budget). Voir `FINDINGS_ladder.md` (vérifié par 6 sceptiques adversariaux ; une prédiction réfutée) et `viz/ladder_interpretation.html`.

### Suite — No Free Lunch plein (panel orthogonal)

L'échelle ci-dessus est emboîtée (linéaire ⊂ MLP) → inversions seulement *vers le bas* (le riche rattrape à compute suffisant). En remplaçant l'échelle de capacité par un **panel de biais orthogonaux** (`inertial`/`affine`/`fourier`/`local`/`fsm` — aucun emboîté), la relationalité devient **irréductible**.

```bash
python orthogonal_experiment.py          # -> results_orthogonal.json
```

Verdict : **NO FREE LUNCH PLEIN**. 7 inversions **bidirectionnelles** budget-robustes, **11 cycles de domination intransitifs** (rock-paper-scissors : `inertial → fourier → affine → inertial`), aucun interprète universel. Aucun budget ne dissout aucune inversion. Voir `FINDINGS_orthogonal.md` et `viz/orthogonal_nfl.html`.

## Prédictions falsifiables

| | attendu si la thèse tient |
|---|---|
| LP(θ) | pique au bord θ_c |
| agent curiosité | se concentre au bord |
| agent erreur | part dans le **chaos** (pas le bord) — discriminateur clé |
| agent aléatoire | plat |
| surrogate (mélange temporel) | aplatit le pic |

Kill-switch central : si l'agent-curiosité et l'agent-erreur se concentrent au même endroit, la curiosité n'ajoute rien à la nouveauté → claim mort.
