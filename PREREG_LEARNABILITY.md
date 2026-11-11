# Pré-enregistrement — La thèse relationnelle, testée directement

*Déclaré AVANT d'exécuter. Suite de l'attaque CML (Addendum FINDINGS 2026-06-10).*
*Date : 2026-06-10. Repo edge-of-curiosity.*

---

## Question

L'Addendum CML a conclu (post-hoc) : *où va la curiosité dépend du couple (substrat × observateur), pas de la criticalité du substrat seule. Le bord ne gagne que pour la lentille qui rend le chaos non-apprenable.* Cette conclusion était une **interprétation**, pas un test. On la teste maintenant directement.

**Mécanisme proposé.** L'erreur de prédiction par rollout croît comme exp(λ·H) où λ = exposant de Lyapunov local et H = horizon de rollout. Donc :
- au **bord** (λ≈0) : prédictible à tout H ;
- dans le **chaos** (λ>0) : prédictible à H court (divergence faible), non-apprenable à H long ;
- dans l'**ordre** (λ<0) : trivialement prédictible → maîtrisé tôt → LP tardif ≈ 0 à tout H.

H est un bouton **de l'observateur seul** (l'horizon qu'on lui demande de prédire), pas du substrat. Le prédicteur et l'entraînement (1-pas) sont inchangés.

## Prédiction décisive (bidirectionnelle, miroir)

| Substrat | bouton | prédiction pré-enregistrée |
|---|---|---|
| **CML** (échoue à H=12 : LP dans le chaos) | H ↑ | argmax(LP) **glisse du chaos vers le bord** quand H croît |
| **Logistique** (passe à H=12 : LP au bord) | H ↓ | argmax(LP) **glisse du bord vers le chaos** quand H décroît |

Variable contrôlée commune : **l'apprenabilité du chaos par l'observateur**, modulée par H seul. Si les deux glissements miroir se produisent, la cible de la curiosité n'est ni le bord ni le chaos en soi — c'est la **frontière d'apprenabilité relative à l'observateur**.

## Critères (pré-enregistrés)

Soit `r_c` le bord (Lyapunov λ=0, **indépendant de H** — vérifié comme sanity check), `d(H) = argmax_r LP(r; H) − r_c` (signé : >0 = côté chaos pour les deux substrats, chaos = r élevé).

| # | Énoncé | KEEP | Kill-switch |
|---|--------|------|-------------|
| **L1** CML monotone | d(H) décroît quand H croît (le pic se rapproche du bord) | Spearman(H, d) ≤ −0.6 sur la grille H | d(H) plat ou croissant → mécanisme faux |
| **L2** CML atteint le bord | à H_max, \|d\| ≤ 2·binw | — | jamais proche du bord → bouton insuffisant |
| **L3** Logistique monotone | d(H) **croît** quand H décroît (pic vers le chaos) | Spearman(H, d) ≥ +0.6 | sens inverse → asymétrie inattendue |
| **L4** Bascule de corrélation | Spearman(LP, criticalité-piquée) passe de − à + quand H croît (CML) et l'inverse (logistique) | changement de signe observé | pas de bascule → pas de contrôle relationnel |
| **L5** Edge H-stable | r_c varie de < 1 binw sur la grille H | — | sinon le repère bouge, test invalide |

## Verdict global (décidé à l'avance)

- **Confirmation forte de la thèse relationnelle** : L1 ∧ L3 (les deux glissements miroir) + L4. → la curiosité cible une frontière d'apprenabilité relationnelle, démontré par un bouton de l'observateur.
- **Confirmation partielle** : un seul des deux glissements (L1 xor L3).
- **Réfutation** : ni L1 ni L3, ou glissements dans le mauvais sens. → la cible de la curiosité serait substrat-intrinsèque après tout, et l'interprétation de l'Addendum était fausse.

## Contrôles

- ≥3 seeds, profil LP moyenné.
- Edge localisé indépendamment (Lyapunov), recalculé par substrat, vérifié H-stable (L5).
- Grille H partagée : {1, 2, 4, 8, 16, 32}.
- (Secondaire, si L1∧L3 passent) : 2e bouton d'observateur — **observation partielle** (k<N sites du CML) — doit reproduire le glissement, confirmant que c'est l'apprenabilité et pas H spécifiquement.
