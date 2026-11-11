"""Substrats dynamiques avec un bouton de controle ordre -> bord -> chaos.

Chaque substrat expose :
  - simulate(theta, steps, seed)        -> trajectoire (steps, dim)
  - criticality_raw(theta, seed)        -> indicateur de criticalite (independant du predicteur)
  - locate_edge(thetas, raw_profile)    -> theta_c (le bord)
  - peaked_from_raw(raw_profile)        -> profil pique au bord (pour correlation)

Trois classes heterogenes pour le test d'universalite (regle des >=3 regimes) :
  continu-1D (logistique), spatio-temporel (CML), discret-booleen (RBN).
"""
import numpy as np


class Substrate:
    name = "base"
    dim = 1
    window = 1            # taille de la fenetre d'entree (plongement par delai)
    out_act = "linear"
    binary = False
    chaos_side = "high"   # cote chaotique du parametre theta : "high" ou "low"

    def __init__(self):
        self.thetas = None
        self.theta_label = "theta"

    def simulate(self, theta, steps, seed, x0=None):
        raise NotImplementedError

    def criticality_raw(self, theta, seed):
        raise NotImplementedError

    def locate_edge(self, thetas, raw):
        raise NotImplementedError

    def peaked_from_raw(self, raw):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
class LogisticMap(Substrate):
    """x_{t+1} = r x_t (1 - x_t). Bord du chaos a r_c ~ 3.5699 (Feigenbaum)."""
    name = "logistic"
    dim = 1
    window = 3
    out_act = "linear"
    chaos_side = "high"   # r eleve = chaos

    def __init__(self, n=30):
        super().__init__()
        self.thetas = np.linspace(3.4, 4.0, n)
        self.theta_label = "r"

    def simulate(self, theta, steps, seed, x0=None):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.1, 0.9) if x0 is None else float(x0)
        for _ in range(200):                       # transitoire
            x = theta * x * (1.0 - x)
        out = np.empty(steps)
        for t in range(steps):
            x = theta * x * (1.0 - x)
            out[t] = x
        return out.reshape(-1, 1)

    def criticality_raw(self, theta, seed):
        # exposant de Lyapunov : moyenne de log|f'(x)|, f'(x) = r(1 - 2x)
        rng = np.random.default_rng(seed + 1)
        x = rng.uniform(0.1, 0.9)
        for _ in range(500):
            x = theta * x * (1 - x)
        s = 0.0
        T = 4000
        for _ in range(T):
            s += np.log(abs(theta * (1 - 2 * x)) + 1e-12)
            x = theta * x * (1 - x)
        return s / T                                # lambda

    def locate_edge(self, thetas, raw):
        lam = np.asarray(raw)
        for i in range(1, len(lam)):                # premier passage lambda<0 -> >=0
            if lam[i - 1] < 0 <= lam[i]:
                t0, t1, l0, l1 = thetas[i - 1], thetas[i], lam[i - 1], lam[i]
                return t0 + (-l0 / (l1 - l0 + 1e-12)) * (t1 - t0)
        return thetas[int(np.argmin(np.abs(lam)))]

    def peaked_from_raw(self, raw):
        return -np.abs(np.asarray(raw))             # max a lambda=0 (le bord)


# --------------------------------------------------------------------------- #
class CoupledMapLattice(Substrate):
    """Reseau de cartes logistiques couplees diffusivement.
    x_i(t+1) = (1-e) f_r(x_i) + (e/2)[f_r(x_{i-1}) + f_r(x_{i+1})].

    Bouton = NONLINEARITE LOCALE r (couplage e FIXE). Donne un vrai axe monotone
    ordre -> bord -> chaos spatio-temporel (analogue de la logistique, en spatial),
    contrairement au sweep en e (transition de synchro = bord flou, refute v1).
    Bord localise par l'exposant de Lyapunov MAXIMAL (lambda_max = 0), pas par la
    susceptibilite de synchro."""
    name = "cml"
    window = 1
    out_act = "linear"
    chaos_side = "high"   # r eleve = chaos spatio-temporel

    def __init__(self, n=26, size=16, eps=0.25):
        super().__init__()
        self.dim = size
        self.eps = eps
        self.thetas = np.linspace(3.4, 4.0, n)
        self.theta_label = "r"

    def _step(self, x, r):
        fx = r * x * (1.0 - x)
        return (1 - self.eps) * fx + 0.5 * self.eps * (np.roll(fx, 1) + np.roll(fx, -1))

    def simulate(self, theta, steps, seed, x0=None):
        rng = np.random.default_rng(seed)
        N = self.dim
        x = rng.uniform(0.1, 0.9, N) if x0 is None else np.array(x0, float)
        for _ in range(250):
            x = self._step(x, theta)
        out = np.empty((steps, N))
        for t in range(steps):
            x = self._step(x, theta)
            out[t] = x
        return out

    def criticality_raw(self, theta, seed):
        # exposant de Lyapunov maximal (Benettin) : croissance d'une perturbation
        # renormalisee a chaque pas. Bord du chaos = lambda_max = 0.
        rng = np.random.default_rng(seed + 1)
        N = self.dim
        x = rng.uniform(0.1, 0.9, N)
        for _ in range(400):
            x = self._step(x, theta)
        d0 = 1e-8
        y = x.copy(); y[0] += d0
        s = 0.0; T = 1600
        for _ in range(T):
            x = self._step(x, theta); y = self._step(y, theta)
            dv = y - x; dist = np.linalg.norm(dv) + 1e-300
            s += np.log(dist / d0)
            y = x + dv * (d0 / dist)                 # renormalisation
        return s / T                                 # lambda_max

    def locate_edge(self, thetas, raw):
        lam = np.asarray(raw)
        for i in range(1, len(lam)):                 # premier passage lambda<0 -> >=0
            if lam[i - 1] < 0 <= lam[i]:
                t0, t1, l0, l1 = thetas[i - 1], thetas[i], lam[i - 1], lam[i]
                return float(t0 + (-l0 / (l1 - l0 + 1e-12)) * (t1 - t0))
        return float(thetas[int(np.argmin(np.abs(lam)))])

    def peaked_from_raw(self, raw):
        return -np.abs(np.asarray(raw))              # max a lambda=0 (le bord)


# --------------------------------------------------------------------------- #
class RandomBooleanNetwork(Substrate):
    """Reseau booleen aleatoire de Kauffman. N noeuds, in-degree K (bouton).
    Bord du chaos a K_c = 2 (p=0.5) : sensibilite de Derrida = 1."""
    name = "rbn"
    window = 1
    out_act = "sigmoid"
    binary = True
    chaos_side = "high"   # K eleve = chaos

    def __init__(self, size=18, ks=(1, 2, 3, 4, 5, 6), p=0.5):
        super().__init__()
        self.dim = size
        self.p = p
        self.thetas = np.array(ks, float)
        self.theta_label = "K"

    def _build(self, K, seed):
        rng = np.random.default_rng(seed)
        N = self.dim
        K = int(round(K))
        inputs = np.array([rng.choice(N, size=K, replace=False) for _ in range(N)])
        tables = (rng.random((N, 2 ** K)) < self.p).astype(np.int8)
        return inputs, tables, K

    def _step(self, state, inputs, tables, K):
        N = self.dim
        idx = np.zeros(N, dtype=int)
        for j in range(K):
            idx = idx * 2 + state[inputs[:, j]]
        return tables[np.arange(N), idx]

    def simulate(self, theta, steps, seed, x0=None):
        inputs, tables, K = self._build(theta, seed)
        rng = np.random.default_rng(seed + 12345)
        N = self.dim
        state = (rng.random(N) < 0.5).astype(np.int8) if x0 is None else np.array(x0, np.int8)
        for _ in range(50):
            state = self._step(state, inputs, tables, K)
        out = np.empty((steps, N), dtype=float)
        for t in range(steps):
            state = self._step(state, inputs, tables, K)
            out[t] = state
        return out

    def criticality_raw(self, theta, seed):
        # sensibilite de Derrida : nb de bits divergents apres 1 pas suite a 1 bit retourne
        inputs, tables, K = self._build(theta, seed)
        rng = np.random.default_rng(seed + 999)
        N = self.dim
        ratios = []
        for _ in range(200):
            s = (rng.random(N) < 0.5).astype(np.int8)
            for _ in range(20):
                s = self._step(s, inputs, tables, K)
            s2 = s.copy()
            s2[rng.integers(N)] ^= 1
            ns = self._step(s, inputs, tables, K)
            ns2 = self._step(s2, inputs, tables, K)
            ratios.append(int(np.sum(ns != ns2)))
        return float(np.mean(ratios))               # ~ K/2 a p=0.5, vaut 1 au bord K=2

    def locate_edge(self, thetas, raw):
        s = np.asarray(raw)
        for i in range(1, len(s)):                  # passage sensibilite < 1 -> >= 1
            if s[i - 1] < 1 <= s[i]:
                t0, t1, s0, s1 = thetas[i - 1], thetas[i], s[i - 1], s[i]
                return t0 + ((1 - s0) / (s1 - s0 + 1e-12)) * (t1 - t0)
        return float(thetas[int(np.argmin(np.abs(s - 1)))])

    def peaked_from_raw(self, raw):
        return -np.abs(np.log(np.asarray(raw) + 1e-9))   # max ou sensibilite = 1 (log = 0)


def all_substrates():
    return [LogisticMap(), CoupledMapLattice(), RandomBooleanNetwork()]
