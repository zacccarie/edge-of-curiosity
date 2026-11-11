"""Panel d'interpretes a biais ORTHOGONAUX — aucun emboite dans un autre.

Contrairement a l'echelle (marginal->...->MLP, emboitee par capacite), ces priors
sont des engagements structurels distincts : aucun n'est un cas particulier d'un
autre, donc aucun ne peut, par montee en capacite, faire le travail d'un autre.
But : obtenir des inversions BIDIRECTIONNELLES biais-fondamentales (No Free Lunch plein).

Interface : fit(Xtr,Ytr,ttr) ; predict(Xte,tte)   (t = temps absolu des cibles).
"""
import numpy as np


class Ortho:
    name = "base"; bias = "—"

    def __init__(self, dim, window, binary=False, budget=1):
        self.dim = dim; self.window = window; self.binary = binary; self.budget = budget

    def _bin(self, v):
        return (v > 0.5).astype(float) if self.binary else v

    def fit(self, Xtr, Ytr, ttr):
        pass

    def predict(self, Xte, tte):
        raise NotImplementedError


class Inertial(Ortho):
    """Prior marche aleatoire : x_hat = dernier etat + derive moyenne. 0 parametre.
    Ne peut PAS osciller (sortie monotone-ish)."""
    name = "inertial"; bias = "marche/inertie"

    def fit(self, Xtr, Ytr, ttr):
        last = Xtr[:, -self.dim:]
        self.drift = (Ytr - last).mean(0)               # derive = premiere difference moyenne

    def predict(self, Xte, tte):
        return self._bin(Xte[:, -self.dim:] + self.drift)


class Affine(Ortho):
    """Prior dynamique lineaire globale (AR moindres carres). Ne peut PAS le XOR."""
    name = "affine"; bias = "lineaire global"

    def fit(self, Xtr, Ytr, ttr):
        A = np.hstack([np.ones((len(Xtr), 1)), Xtr])
        self.W, *_ = np.linalg.lstsq(A, Ytr, rcond=None)

    def predict(self, Xte, tte):
        return self._bin(np.hstack([np.ones((len(Xte), 1)), Xte]) @ self.W)


class Fourier(Ortho):
    """Prior harmonique global : somme de K sinusoides, fonction du TEMPS (pas de l'etat).
    Vit dans la frequence. Ne peut PAS suivre un processus sans frequence stable (marche)."""
    name = "fourier"; bias = "harmonique"

    def _refine(self, f0, t, y, n, refine):
        """Raffine une frequence FFT (resolution 1/n trop grossiere pour extrapoler loin)."""
        if not refine:
            return f0
        lo, hi = max(1e-6, f0 - 1.5 / n), f0 + 1.5 / n
        best, besterr = f0, np.inf
        for f in np.linspace(lo, hi, 40):
            D = np.vstack([np.ones_like(t), t, np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]).T
            w, *_ = np.linalg.lstsq(D, y, rcond=None)
            e = float(np.sum((D @ w - y) ** 2))
            if e < besterr:
                besterr, best = e, f
        return best

    def fit(self, Xtr, Ytr, ttr):
        self.t0 = ttr[0]
        K = int(self.budget)
        refine = (self.dim == 1)                          # raffinement coûteux : foyer = 1D
        self.models = []
        for j in range(self.dim):
            y = Ytr[:, j].astype(float)
            t = (ttr - self.t0).astype(float)
            G = np.vstack([np.ones_like(t), t]).T         # tendance lineaire retiree
            coef, *_ = np.linalg.lstsq(G, y, rcond=None)
            r = y - G @ coef
            n = len(r)
            fft = np.abs(np.fft.rfft(r * np.hanning(n)))
            freqs = np.fft.rfftfreq(n)
            order = np.argsort(-fft)
            picked = []
            for idx in order:
                if freqs[idx] > 1e-6:
                    picked.append(self._refine(freqs[idx], t, y, n, refine))
                if len(picked) >= K:
                    break
            cols = [np.ones_like(t), t]
            for f in picked:
                cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
            D = np.vstack(cols).T
            w, *_ = np.linalg.lstsq(D, y, rcond=None)
            self.models.append((picked, w))

    def predict(self, Xte, tte):
        t = (tte - self.t0).astype(float)
        out = np.empty((len(tte), self.dim))
        for j, (picked, w) in enumerate(self.models):
            cols = [np.ones_like(t), t]
            for f in picked:
                cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
            out[:, j] = np.vstack(cols).T @ w
        return self._bin(out)


class Local(Ortho):
    """Prior de localite : successeur moyen des k contextes voisins (non-param).
    Ne peut PAS extrapoler hors du support echantillonne."""
    name = "local"; bias = "localite"

    def fit(self, Xtr, Ytr, ttr):
        self.Xtr = Xtr; self.Ytr = Ytr
        self.k = min(max(1, int(self.budget)), len(Xtr))

    def predict(self, Xte, tte):
        d2 = ((Xte[:, None, :] - self.Xtr[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d2, self.k - 1, axis=1)[:, :self.k]
        return self._bin(self.Ytr[idx].mean(1))


class FSM(Ortho):
    """Prior machine a etats discrete : moyenne des successeurs par MOTIF binarise de
    la fenetre (table). Vit dans le motif discret. Ne peut PAS lire l'amplitude continue."""
    name = "fsm"; bias = "motif discret"

    def fit(self, Xtr, Ytr, ttr):
        self.med = np.median(Xtr, axis=0)
        keys = self._keys(Xtr)
        self.table = {}
        for k, y in zip(keys, Ytr):
            self.table.setdefault(k, []).append(y)
        self.table = {k: np.mean(v, 0) for k, v in self.table.items()}
        self.default = Ytr.mean(0)

    def _keys(self, X):
        b = (X > self.med).astype(np.int8)               # binarisation par mediane d'entrainement
        return [tuple(row) for row in b]

    def predict(self, Xte, tte):
        out = np.array([self.table.get(k, self.default) for k in self._keys(Xte)])
        return self._bin(out)


def make_panel(dim, window, binary, budgets=None):
    """Budgets par defaut (capacite « normale »). Le bras budget les pousse au max."""
    b = budgets or {}
    return [
        Inertial(dim, window, binary),
        Affine(dim, window, binary),
        Fourier(dim, window, binary, budget=b.get("fourier", 6)),
        Local(dim, window, binary, budget=b.get("local", 5)),
        FSM(dim, window, binary),
    ]
