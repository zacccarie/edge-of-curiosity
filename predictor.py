"""Predicteur a capacite bornee (petit MLP numpy) + skill de rollout + learning progress.

La capacite est volontairement limitee : un predicteur trop puissant apprendrait
meme le chaos, et l'experience perdrait son sens. Le skill se mesure en rollout
multi-pas (free-running), tenu a l'ecart des donnees d'entrainement.
"""
import numpy as np


class MLP:
    """Une couche cachee, retropropagation manuelle, SGD."""

    def __init__(self, in_dim, hid, out_dim, lr=0.05, seed=0, out_act="linear"):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0 / np.sqrt(in_dim), (in_dim, hid))
        self.b1 = np.zeros(hid)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(hid), (hid, out_dim))
        self.b2 = np.zeros(out_dim)
        self.lr = lr
        self.out_act = out_act

    def forward(self, X):
        a1 = np.tanh(X @ self.W1 + self.b1)
        z2 = a1 @ self.W2 + self.b2
        out = 1.0 / (1.0 + np.exp(-z2)) if self.out_act == "sigmoid" else z2
        return out, a1

    def train_step(self, X, Y):
        B = X.shape[0]
        out, a1 = self.forward(X)
        if self.out_act == "sigmoid":
            dz2 = (out - Y) * out * (1 - out) / B
        else:
            dz2 = (out - Y) / B
        dW2 = a1.T @ dz2
        db2 = dz2.sum(0)
        dz1 = (dz2 @ self.W2.T) * (1 - a1 ** 2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(0)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


def make_windows(traj, d):
    """(fenetre de d pas aplatie) -> etat suivant."""
    T = traj.shape[0]
    X, Y = [], []
    for t in range(d, T - 1):
        X.append(traj[t - d:t].reshape(-1))
        Y.append(traj[t])
    return np.array(X), np.array(Y)


FLOOR = 0.05   # plancher : sous une NMSE de 5%, le skill sature (anti "anti-noisy-TV")


def skill_of(nmse):
    """Skill borne a partir de la NMSE. Le plancher empeche le regime ordonne
    (NMSE -> 0) de rafler un progres infini via -log(MSE)."""
    return float(-np.log(nmse + FLOOR))


def _apply_lens(A, lens):
    """Transforme un lot d'etats (rows = vecteurs spatiaux) selon la lentille de
    l'observateur. 'full' = etat complet ; 'coarse' = champ moyen (mode collectif
    lent) ; 'fine' = deviation au champ moyen (structure turbulente haute-frequence)."""
    if lens == "full" or A.shape[1] == 1:
        return A
    if lens == "coarse":
        return A.mean(axis=1, keepdims=True)
    if lens == "fine":
        return A - A.mean(axis=1, keepdims=True)
    raise ValueError(lens)


def rollout_nmse(mlp, sub, theta, seed, H=12, n_init=8, shuffle=False, lens="full"):
    """Skill de rollout : free-run sur H pas depuis des CI tenues a l'ecart.
    Renvoie la MSE NORMALISEE par la variance du signal (comparable entre regimes).
    shuffle=True -> controle surrogate. lens -> observable evalue (full/coarse/fine)."""
    d = sub.window
    preds_all, trues_all = [], []
    for k in range(n_init):
        tr = sub.simulate(theta, H + d + 2, seed=seed * 1000 + k + 1)
        if shuffle:                                   # detruit la structure temporelle
            rng = np.random.default_rng(seed * 50 + k)
            tr = tr[rng.permutation(len(tr))]
        win = tr[:d].copy()
        preds = []
        for _ in range(H):
            out, _ = mlp.forward(win.reshape(1, -1))
            nxt = out.reshape(sub.dim)
            if sub.binary:
                nxt = (nxt > 0.5).astype(float)
            preds.append(nxt)
            win = np.vstack([win[1:], nxt]) if d > 1 else nxt.reshape(1, -1)
        preds = np.array(preds)
        true = tr[d:d + H]
        m = min(len(preds), len(true))
        preds_all.append(preds[:m])
        trues_all.append(true[:m])
    P = _apply_lens(np.concatenate(preds_all), lens)
    T = _apply_lens(np.concatenate(trues_all), lens)
    mse = float(np.mean((P - T) ** 2))
    var = float(np.var(T)) + 1e-6                      # variance du signal (normalisation)
    return mse / var


def intrinsic_lp(sub, theta, seed, hid=16, lr=0.05, budget=1200, chunk=150,
                 H=12, n_init=8, lp_start_frac=0.25, shuffle=False, lens="full"):
    """LP intrinseque = amelioration TARDIVE du skill de rollout.

    Predicteur frais, entraine `budget` pas en mini-lots. On echantillonne le
    skill tous les `chunk` pas. LP = skill(fin) - skill(premier quart).
    Annule l'ordre (deja maitrise tot) et le chaos (jamais maitrise)."""
    d = sub.window
    mlp = MLP(d * sub.dim, hid, sub.dim, lr=lr, seed=seed, out_act=sub.out_act)
    train_traj = sub.simulate(theta, budget + d + 50, seed=seed * 7 + 3)
    if shuffle:
        rng0 = np.random.default_rng(seed * 91 + 2)
        train_traj = train_traj[rng0.permutation(len(train_traj))]
    Xtr, Ytr = make_windows(train_traj, d)
    n = len(Xtr)
    rng = np.random.default_rng(seed * 13 + 5)
    curve = []
    step = 0
    while step < budget:
        idx = rng.integers(0, n, size=min(32, n))
        mlp.train_step(Xtr[idx], Ytr[idx])
        step += 1
        if step % chunk == 0:
            nmse = rollout_nmse(mlp, sub, theta, seed=seed + 101, H=H, n_init=n_init,
                                shuffle=shuffle, lens=lens)
            curve.append(skill_of(nmse))              # skill borne, NMSE normalisee
    curve = np.array(curve)
    if len(curve) < 4:
        return 0.0, curve
    q = max(1, int(len(curve) * lp_start_frac))
    lp = float(curve[-2:].mean() - curve[q - 1:q + 1].mean())
    return lp, curve
