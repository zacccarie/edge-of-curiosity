"""Agents d'allocation d'attention sur l'espace des regimes theta.

Trois moteurs :
  - "lp"     : curiosite = progres d'apprentissage (Schmidhuber/Oudeyer)
  - "error"  : nouveaute = erreur brute (le piege "tele qui neige")
  - "random" : controle uniforme

Un predicteur PERSISTANT par bin theta. A chaque pull : choisir un bin,
l'entrainer un peu, observer une recompense, mettre a jour la politique.
La sortie est la distribution de sejour (dwell) en phase tardive : ou la
curiosite finit-elle par se concentrer une fois les regimes faciles maitrises ?
"""
import numpy as np
from predictor import MLP, make_windows, rollout_nmse, skill_of


def _softmax_choice(score, temp, rng, eps=0.1):
    A = len(score)
    if rng.random() < eps:
        return int(rng.integers(A))
    s = np.asarray(score, float)
    s = s - s.max()
    p = np.exp(s / max(temp, 1e-3))
    p /= p.sum()
    return int(rng.choice(A, p=p))


def run_agent(sub, mode, seed, hid=16, lr=0.05, pulls=None, chunk=20,
              H=12, n_init=10, temp=0.5):
    thetas = sub.thetas
    A = len(thetas)
    if pulls is None:
        pulls = A * 25
    d = sub.window
    rng = np.random.default_rng(seed * 31 + 7)
    mlps = [MLP(d * sub.dim, hid, sub.dim, lr=lr, seed=seed * 100 + a, out_act=sub.out_act)
            for a in range(A)]
    data = [make_windows(sub.simulate(thetas[a], pulls + d + 50, seed=seed * 7 + a + 1), d)
            for a in range(A)]
    # LP lisse par double-EMA du skill (rapide - lente). PAS de rectification
    # par echantillon : sur un signal bruite mais stationnaire (chaos), rapide et
    # lente suivent la meme moyenne -> LP ~ 0 (le bruit s'annule au lieu d'etre
    # rectifie en faux progres). Le bord (skill qui monte) -> rapide > lente -> LP > 0.
    skill_fast = [None] * A
    skill_slow = [None] * A
    lp_signal = np.zeros(A)
    skill_cur = np.full(A, np.nan)
    history = []
    for _ in range(pulls):
        if mode == "random":
            a = int(rng.integers(A))
        elif mode == "lp":
            a = _softmax_choice(lp_signal, temp, rng, eps=0.1)
        elif mode == "error":                          # cherche l'erreur max -> chaos
            err = np.array([(-skill_cur[i] if not np.isnan(skill_cur[i]) else 5.0)
                            for i in range(A)])
            a = _softmax_choice(err, temp, rng, eps=0.1)
        else:
            raise ValueError(mode)

        X, Y = data[a]
        nn = len(X)
        for _ in range(chunk):
            idx = rng.integers(0, nn, size=min(32, nn))
            mlps[a].train_step(X[idx], Y[idx])
        skill = skill_of(rollout_nmse(mlps[a], sub, thetas[a], seed=seed + 202,
                                      H=H, n_init=n_init))
        skill_cur[a] = skill
        if skill_fast[a] is None:
            skill_fast[a] = skill_slow[a] = skill
        else:
            skill_fast[a] = 0.5 * skill_fast[a] + 0.5 * skill
            skill_slow[a] = 0.85 * skill_slow[a] + 0.15 * skill
        lp_signal[a] = skill_fast[a] - skill_slow[a]   # progres lisse, signe
        history.append(a)

    history = np.array(history)
    late = history[int(0.7 * len(history)):]
    dwell = np.bincount(late, minlength=A).astype(float)
    dwell /= dwell.sum()
    return dwell
