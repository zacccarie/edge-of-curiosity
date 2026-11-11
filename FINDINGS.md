# FINDINGS — La curiosité cherche-t-elle le bord du chaos ?

*2026-06-10. Résultats honnêtes, verdict pré-enregistré, deux artefacts attrapés en chemin.*
*Voir `PREREGISTRATION.md` (déclaré avant les runs) et `results_v3.json` (données).*

---

## Verdict en une phrase

**Conjecture « argmax(progrès d'apprentissage) = bord du chaos » : confirmée sur la seule logistique (1/3), robustement réfutée sur CML et RBN où le LP pique DANS le chaos.** Au critère strict pré-enregistré (H1∧H3), 1/3 passe. La leçon finale n'est pas « périmètre continu/discret » (réfuté, voir Addendum) mais **relationnelle** : *où va la curiosité dépend du couple (substrat × prédicteur), pas de la criticalité du substrat seule* — le bord ne l'emporte que pour la lentille qui rend le chaos non-apprenable (logistique basse-dim). C'est le finding-maître **ρ(X,M)** reconfirmé par l'expérience censée le contredire.

## Ce qui était testé

*argmax(progrès d'apprentissage) coïncide-t-il avec le bord du chaos ?* — le chaînon manquant entre une mesure passive du bord (β, Γ dans le corpus) et un **agent qui le vise**. Dispositif : prédicteur à capacité bornée, skill de rollout multi-pas, LP = amélioration tardive ; 3 agents (curiosité / nouveauté / aléatoire) ; 3 substrats hétérogènes.

## Résultats par substrat (run v3, métrique + agent corrigés)

| Substrat | LP pique au bord (H1) | agent curiosité → bord | agent erreur → chaos (H3) | random plat (H4) | Spearman (H5) | surrogate propre | **H6** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistique** (r) | ✅ 3.545 / bord 3.571 | ✅ 3.648 (bande du bord) | ✅ → 4.0 | ✅ | ✅ +0.51 | ✅ | **PASS** |
| **CML** (ε) | ❌ 0.121 (LP élevé au bord mais argmax ailleurs) | ~ 0.096 | ✅ → 0.17 | ✅ | ❌ +0.34 | ✅ | FAIL |
| **RBN** (K) | ❌ pique en chaos (K=4,5) | ❌ → K=1 | ❌ → K=1 | ✅ | ❌ −0.60 | — | FAIL |

**Logistique = confirmation pleine** : sur le substrat à transition la plus nette (Feigenbaum r_c≈3.57), les cinq prédictions tiennent — LP intrinsèque pique au bord, l'agent curieux s'y installe, l'agent-nouveauté part dans le chaos (discriminateur), l'aléatoire est plat, le surrogate est propre.

## Vérification adversariale du positif (robustesse)

Le pic de LP au bord (logistique) survit-il aux choix de réglage, ou est-ce un artefact ? Balayage FLOOR × capacité (`robustness.py`) :

- **argmax(LP) au bord dans 10/12 configurations** — toutes celles avec FLOOR ≥ 0.05, à travers hid ∈ {12,16,24}.
- Les **2 seuls échecs** : FLOOR=0.02 (plancher trop bas) → le régime ordonné reprend le dessus. C'est exactement l'artefact diagnostiqué (cf. ci-dessous) ; il ne revient que si on désactive le garde-fou.
- Spearman positif dans **12/12** configs (+0.08 à +0.72), s'affaiblissant à haute capacité (hid=24) — cohérent : le mécanisme exige un prédicteur **à capacité bornée** (pré-enregistré). Le résultat n'est pas un réglage chanceux.

## Deux artefacts attrapés en chemin (la méthode comme livrable)

Le cycle « hypothèse → expérience conçue pour tuer → diagnostic de l'échec → marche suivante » a tourné deux fois :

1. **Métrique confondue (run v1 → v2).** skill = −log(MSE brute) : (a) MSE non normalisée → le régime ordonné, à variance de signal minuscule, a une MSE minuscule indépendamment de tout apprentissage ; (b) −log non borné → MSE→0 fait exploser le skill, et des gains triviaux dans l'ordre dominent le LP. → argmax(LP) tombait dans l'ordre. **Fix** : NMSE normalisée par la variance du signal + plancher (skill borné). Le pic est passé au bord.

2. **Rectification du bruit dans le bandit (run v2 → v3).** La récompense `max(skill−skill_prev, 0)` rectifiait le bruit d'estimation en faux progrès → les régimes à forte variance (chaos) accumulaient du LP fictif → l'agent curieux dérivait vers le chaos, au même endroit que l'agent-nouveauté. **Fix** : LP lissé par double-EMA (rapide − lente), sans rectification par échantillon → le bruit stationnaire s'annule au lieu de s'accumuler. L'agent curieux s'est replacé sur le bord.

Aucune hypothèse ni aucun seuil pré-enregistré n'a été modifié — seulement deux implémentations de mesure clairement biaisées.

## Le périmètre — version corrigée (voir Addendum 2026-06-10)

> ⚠️ La première lecture ci-dessous (« continu vs discret ») a été **réfutée** par l'attaque CML.
> Le découpage correct est **relationnel**, pas substrat-intrinsèque. Lire l'Addendum.

Première lecture (incorrecte, conservée pour la trace) : *au bord, le rollout serait maximalement apprenable*, le mécanisme supposant une prédictibilité régime-dépendante ; d'où l'hypothèse « continu sensible (logistique/CML) confirme, discret (RBN) brise ». L'attaque CML (r-sweep + Lyapunov + dose-réponse capacité) a montré que **le CML se comporte comme le RBN, pas comme la logistique** — donc le clivage continu/discret est faux. Voir Addendum.

## Lien à la thèse des essais

La thèse « l'intelligence est un équilibre activement tenu au bord d'une instabilité » reçoit une assise empirique **partielle et lentille-dépendante** : sur la logistique, un agent curieux cherche le bord là où la nouveauté brute part dans le chaos. Mais cela ne se généralise PAS (CML, RBN : la curiosité va au chaos). La leçon n'est donc pas « le bord du chaos attire la curiosité » mais « ce que la curiosité trouve est relationnel » (voir Addendum + finding ρ(X,M)).

## Limites honnêtes / prochaines marches

- 3 substrats, 5 seeds, petits modèles. La logistique est nette ; CML et RBN demandent un meilleur instrument.
- **CML** : bord large/multiple via la susceptibilité de synchronisation ; tester un bouton plus net (r à ε fixe) ou un locateur de bord par Lyapunov transverse.
- **RBN** : tester avec une tâche où la prédictibilité s'effondre vraiment en chaos (observation partielle / horizon long), ou un balayage continu de p — pour voir si le contre-résultat est dû au substrat ou à la tâche de prédiction.
- **H2** (concentration de l'agent au bord) reste sous le critère strict même sur logistique : l'agent s'installe dans la *bande* du bord, pas sur un bin unique. Bandit plus directif (UCB sur LP lissé) à tester.
- Le pont reste à porter sur un vrai réseau de neurones (le substrat = l'architecture qui s'auto-modifie, adaptive-nn Phase 2).

---

## Addendum 2026-06-10 — attaque CML, dose-réponse capacité, et correction du périmètre

**But** : tenter le 2/3 en donnant au CML un bord net (bouton = `r` local à couplage ε=0.25 fixe ; bord = Lyapunov maximal λ=0), le vrai analogue spatio-temporel du test logistique. Données : `results_cml_r.json`, `cml_capacity.log`.

**Résultat CML r-sweep (5 seeds)** : bord λ=0 à r_c=3.580 (propre). Mais LP intrinsèque pique à **r=3.88 (plein chaos)**, **Spearman(LP, criticalité) = −0.71**, agent-erreur → 3.52 (côté ordre). **H6 FAIL.** Le CML ne se comporte PAS comme la logistique — il se comporte comme le RBN : LP dans le chaos, anti-corrélé au bord.

**Test mécanistique (la capacité explique-t-elle l'échec ?)** — hypothèse : « LP-pique-au-bord exige une capacité suffisante pour maîtriser le bord mais pas le chaos ». Dose-réponse hid ∈ {16, 48, 96, 160} :

| hid | argmax LP | dist au bord | Spearman |
|----:|:---:|:---:|:---:|
| 16 | 3.880 | 0.301 | −0.65 |
| 48 | 3.784 | 0.205 | −0.60 |
| 96 | 3.784 | 0.205 | −0.63 |
| 160 | 3.784 | 0.205 | −0.67 |

→ Multiplier la capacité par 10 **ne déplace pas** le pic vers le bord (plateau à r=3.78, dist ≫ tol 0.036) ; Spearman reste fortement négatif partout. **Hypothèse capacité RÉFUTÉE.** L'attraction de la curiosité vers le chaos CML est robuste, pas un artefact de sous-capacité.

**Correction du périmètre.** Le clivage « continu (confirme) vs discret (brise) » de la v1 est **faux** : le CML est continu et échoue comme le RBN discret. Le vrai facteur n'est pas le substrat seul.

**Conclusion (relationnelle, alignée ρ(X,M)).** Où la curiosité se pose **dépend du couple (substrat × prédicteur)**, pas de la criticalité du substrat. Le bord ne l'emporte que pour la lentille qui rend le chaos *génuinement non-apprenable* — la logistique 1D, où la barrière de Lyapunov sature le rollout sur un signal de basse dimension. Dès que le chaos offre une structure réductible persistante (CML couplé, RBN à tables fixes), la curiosité — et l'agent — y vont. **La frontière de compétence est relationnelle, pas une propriété intrinsèque du bord du chaos.** L'expérience conçue pour confirmer la thèse romantique reconfirme à la place le finding-maître ρ(X,M) : *l'intrinsèque ne prédit pas le relationnel* (n-ième occurrence).

**Statut de la thèse des essais.** « Intelligence = équilibre tenu au bord » survit comme *intuition* mais perd son universalité opérationnelle : la curiosité ne cherche pas le bord du chaos en général. Reformulation honnête : la curiosité cherche **la zone de progrès de compression maximal**, qui *coïncide* avec le bord du chaos seulement quand l'observateur ne peut rien apprendre du chaos. Bord du chaos ⊂ frontière d'apprenabilité relationnelle, pas l'inverse.

---

## Addendum 2 — Test direct de la thèse relationnelle (horizon + lentille)

`PREREG_LEARNABILITY.md` · données `results_learnability.json`, `results_lens_cml.json`.

**But.** Tester *directement* la conclusion relationnelle de l'Addendum 1 : si la cible de la curiosité est une propriété du couple (substrat × observateur), un bouton de l'observateur seul doit la déplacer. Bouton principal = **horizon de rollout H** (l'erreur croît en exp(λ·H), donc H module l'apprenabilité du chaos). Prédiction miroir : pic LP glisse vers le bord quand H ↑.

**Logistique — CONFIRMÉ, proprement.** H : 1 → 32 ⇒ argmax(LP) glisse en continu **chaos → bord → ordre** (3.94 → 3.40), corrélation(LP, criticalité) **change de signe** (−0.39 → +0.71), Spearman(H, d) = **−0.99**. Un bouton de l'observateur seul relocalise la cible de la curiosité sur substrat fixe. C'est la démonstration directe de ρ(X,M).

**CML — le bouton ne mord pas.** Le pic reste collé dans le chaos (3.88) à **tout H** (Spearman(H,d)=0.00) ; l'anti-corrélation faiblit (−0.75 → −0.46) mais le pic ne bouge pas. Hypothèse « modes collectifs lents apprenables » testée par une lentille fine/grossière (predire le champ moyen vs la déviation) :

| lentille CML (H=16) | argmax LP | cible | Spearman |
|---|:---:|:---:|:---:|
| grossière (champ moyen) | 3.832 | chaos | −0.35 |
| complète | 3.880 | chaos | −0.62 |
| fine (turbulence) | 3.880 | chaos | −0.73 |

→ La lentille fine **ne déplace pas** le pic (reste chaos, anti-corr renforcée). **Hypothèse modes-grossiers réfutée.** Trois boutons d'observateur indépendants (horizon, capacité Addendum 1, lentille) échouent tous à décoller la curiosité du chaos CML.

**Conclusion finale (robuste).** La thèse romantique « curiosité → bord du chaos » est **réfutée en général**. Remplacement validé :
> La curiosité cherche la **zone de structure réductible maximale**. Sa localisation est une propriété du couple (substrat × observateur) — *lens-contrôlable* quand le chaos peut être rendu non-apprenable (logistique 1D : démontré par le glissement en H), *non relocalisable* quand le chaos offre une structure persistante au prédicteur borné (CML 16D : robuste sur 3 boutons). Le bord du chaos n'est qu'un **cas particulier** de cette frontière d'apprenabilité, pas son lieu universel.

Le cœur relationnel ρ(X,M) sort **renforcé** (un bouton d'observateur relocalise la cible — logistique) ; l'universalité du bord du chaos sort **réfutée** (CML). Cohérent avec `pont_maitre_finding` : l'intrinsèque (criticalité du substrat) ne prédit pas le relationnel (où la curiosité se pose).
