# Pré-enregistrement — La curiosité cherche-t-elle le bord du chaos ?

*Déclaré AVANT d'exécuter l'expérience. Aucun seuil ne sera modifié après avoir vu les résultats.*
*Date : 2026-06-10. Programme : edge-of-curiosity (adaptive-nn Phase 2 — le chaînon manquant).*

---

## Question décisive

Un agent guidé par une **curiosité endogène** (progrès d'apprentissage / *learning progress*) concentre-t-il son exploration **au bord du chaos** — c'est-à-dire au régime critique θ_c où la susceptibilité dynamique est maximale ?

C'est la conjecture maîtresse jamais testée des deux essais : *argmax(progrès d'apprentissage) coïncide-t-il avec le pic de susceptibilité critique ?* Si oui, la thèse « l'intelligence est un équilibre activement tenu au bord d'une instabilité » reçoit une assise empirique. Si non, le bord du chaos est une corniche subie, pas visée — et la thèse tombe proprement.

## Dispositif

- **3 substrats hétérogènes** (règle des ≥3 régimes) : carte logistique (bouton `r`, continu-1D), réseau de cartes couplées CML (bouton `ε`, spatio-temporel), réseau booléen aléatoire RBN (bouton `K`, discret). Chacun a un bord du chaos analytiquement connu (logistique r_c≈3.57 ; CML transition de synchronisation ; RBN K_c=2).
- **Tâche** : prédiction par rollout multi-pas (free-running sur H pas), mesurée sur des conditions initiales tenues à l'écart. Aligné dynamoscope / CKA (T_pred).
- **Prédicteur à capacité bornée** (petit MLP). Capacité volontairement limitée : sinon le chaos serait apprenable et l'expérience perdrait son sens.
- **Progrès d'apprentissage intrinsèque** LP(θ) = amélioration *tardive* du skill de rollout (skill final − skill au premier quart). Annule les deux extrêmes : l'ordre (vite maîtrisé → LP tardif nul) et le chaos (jamais maîtrisé → LP nul).
- **3 agents** allouant un budget de pulls sur les bins θ : `lp` (curiosité = progrès d'apprentissage), `error` (nouveauté = erreur brute, le piège « télé qui neige »), `random` (contrôle).

## Hypothèses et seuils (pré-enregistrés)

| # | Énoncé | Critère KEEP | Kill-switch |
|---|--------|--------------|-------------|
| **H1** intrinsèque | LP(θ) pique au bord | \|argmax LP − θ_c\| ≤ tol | LP pique en plein chaos → « curiosité = bord » FAUX |
| **H2** comportemental | L'agent-LP se concentre au bord | \|argmax dwell_LP − θ_c\| ≤ tol ET dwell concentré (entropie < 0.9× uniforme) | agent-LP plat ou ailleurs |
| **H3** discriminateur | L'agent-erreur va au CHAOS, pas au bord | argmax dwell_error du côté chaotique de θ_c, ET bin ≠ agent-LP | agent-LP et agent-erreur au même endroit → la curiosité n'ajoute rien à la nouveauté → **claim central mort** |
| **H4** contrôle | L'agent aléatoire est plat | max(dwell_random) < 1.5/A | sinon bug de bandit |
| **H5** corrélation | LP suit la criticalité | Spearman(LP, criticalité-piquée) ≥ 0.5 | — |
| **H6** universalité | H1 ∧ H3 tiennent | 3/3 substrats = fort ; 2/3 = partiel ; ≤1/3 = thèse affaiblie | — |

- `tol` = 1.5 × largeur de bin (continu) ; ±1 en unités de K (RBN).
- `θ_c` est localisé **indépendamment** du prédicteur (Lyapunov λ=0 pour logistique/CML-via-susceptibilité ; sensibilité de Derrida=1 pour RBN).

## Contrôle anti-artefact (surrogate)

Mélanger temporellement la trajectoire du substrat (détruire la structure dynamique, préserver les marginales) doit **aplatir** le profil LP et déplacer/effacer le pic. Si le surrogate reproduit le pic au bord → c'est un artefact, pas la dynamique → interprétation tuée.

## Verdict global (décidé à l'avance)

- **Confirmation forte** : H1 ∧ H3 sur 3/3 substrats + surrogate plat. → la curiosité cherche bien le bord du chaos.
- **Confirmation partielle** : 2/3. → effet réel, périmètre à resserrer.
- **Réfutation** : ≤1/3, ou agent-LP = agent-erreur, ou surrogate reproduit le pic. → la thèse « équilibre activement tenu » est fausse ou non démontrée — et on le dit.
