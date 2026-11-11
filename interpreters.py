"""Echelle d'interpretes — un primitif inne ajoute par barreau.

Interface uniforme : chaque interprete sait
  - fit(Xtr, Ytr)            : apprendre sur le split d'entrainement
  - predict_onestep(Xte)     : prediction un-pas teacher-forced (held-out)
  - step(win_flat) -> next   : un pas libre (pour le rollout / Gamma)

Le plongement par delai (fenetre D) est commun ; chaque barreau n'exploite que
ce que son primitif permet (la persistance ne regarde que le dernier etat, la
marginale aucun). On teste si l'ordre de ces biais est total (intrinseque) ou
admet des inversions selon la classe de donnees (relationnel).
"""
import numpy as np

from predictor import MLP


class Interpreter:
    level = -1
    name = "base"
    primitive = "—"

    def __init__(self, dim, window, binary=False):
        self.dim = dim
        self.window = window
        self.binary = binary

    def fit(self, Xtr, Ytr):
        pass

    def predict_onestep(self, Xte):
        raise NotImplementedError

    def step(self, win_flat):
        raise NotImplementedError

    def _binarize(self, v):
        return (v > 0.5).astype(float) if self.binary else v


# --------------------------------------------------------------------------- #
class Marginal(Interpreter):
    """L0 — aucune memoire. Predit la moyenne marginale. Plancher."""
    level = 0; name = "marginal"; primitive = "—"

    def fit(self, Xtr, Ytr):
        self.mu = Ytr.mean(0)

    def predict_onestep(self, Xte):
        return np.tile(self.mu, (len(Xte), 1))

    def step(self, win_flat):
        return self.mu.copy()


class Persistence(Interpreter):
    """L1 — memoire du dernier etat. x_hat_{t+1} = x_t."""
    level = 1; name = "persistence"; primitive = "memoire"

    def _last(self, X):
        return X[:, -self.dim:]                          # dernier bloc de la fenetre

    def predict_onestep(self, Xte):
        return self._last(Xte)

    def step(self, win_flat):
        return win_flat[-self.dim:].copy()


class KNN(Interpreter):
    """L2 — memoire + comparaison. Moyenne des successeurs des k contextes les plus
    proches (non-parametrique, biais de localite)."""
    level = 2; name = "knn"; primitive = "comparaison"

    def __init__(self, dim, window, binary=False, k=5):
        super().__init__(dim, window, binary); self.k = k

    def fit(self, Xtr, Ytr):
        self.Xtr = Xtr; self.Ytr = Ytr
        self.k_eff = min(self.k, len(Xtr))

    def _query(self, Q):
        # distances Q(n,d) x Xtr(m,d) -> (n,m), moyenne des k plus proches successeurs
        d2 = ((Q[:, None, :] - self.Xtr[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d2, self.k_eff - 1, axis=1)[:, :self.k_eff]
        return self.Ytr[idx].mean(1)

    def predict_onestep(self, Xte):
        return self._query(Xte)

    def step(self, win_flat):
        return self._binarize(self._query(win_flat[None, :])[0])


class Linear(Interpreter):
    """L3 — regle parametrique globale. Moindres carres affine sur la fenetre."""
    level = 3; name = "linear"; primitive = "regle globale"

    def fit(self, Xtr, Ytr):
        A = np.hstack([np.ones((len(Xtr), 1)), Xtr])     # biais + fenetre
        self.W, *_ = np.linalg.lstsq(A, Ytr, rcond=None)

    def predict_onestep(self, Xte):
        return np.hstack([np.ones((len(Xte), 1)), Xte]) @ self.W

    def step(self, win_flat):
        out = np.hstack([[1.0], win_flat]) @ self.W
        return self._binarize(out)


class MLPInterp(Interpreter):
    """L4/L5 — non-linearite bornee (petit MLP) puis capacite (large)."""

    def __init__(self, dim, window, binary=False, hid=8, budget=1500, lr=0.05,
                 seed=0, level=4, name="mlp_small"):
        super().__init__(dim, window, binary)
        self.hid = hid; self.budget = budget; self.lr = lr; self.seed = seed
        self.level = level; self.name = name
        self.primitive = "non-linearite" if level == 4 else "capacite"

    def fit(self, Xtr, Ytr):
        self.net = MLP(Xtr.shape[1], self.hid, self.dim, lr=self.lr, seed=self.seed,
                       out_act="sigmoid" if self.binary else "linear")
        rng = np.random.default_rng(self.seed * 13 + 5)
        n = len(Xtr)
        for _ in range(self.budget):
            idx = rng.integers(0, n, size=min(32, n))
            self.net.train_step(Xtr[idx], Ytr[idx])

    def predict_onestep(self, Xte):
        out, _ = self.net.forward(Xte)
        return out

    def step(self, win_flat):
        out, _ = self.net.forward(win_flat[None, :])
        return self._binarize(out[0])


# --------------------------------------------------------------------------- #
def make_ladder(dim, window, binary, seed):
    """Les six barreaux, structure innee croissante."""
    return [
        Marginal(dim, window, binary),
        Persistence(dim, window, binary),
        KNN(dim, window, binary, k=5),
        Linear(dim, window, binary),
        MLPInterp(dim, window, binary, hid=8, budget=1500, seed=seed, level=4, name="mlp_small"),
        MLPInterp(dim, window, binary, hid=32, budget=2500, seed=seed, level=5, name="mlp_large"),
    ]
