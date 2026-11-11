"""Classes de flux a structure heterogene — l'axe de variation de l'experience.

Chaque generateur expose :
  - generate(steps, seed) -> trajectoire (steps, dim), float
  - binary : bool (sortie 0/1)
  - kind   : etiquette de famille structurelle

L'objectif n'est PAS de balayer un parametre interne (comme substrates.py) mais
de couvrir des *types* de structure differents : aucune / periodique / lineaire
court / integre / lineaire long / non-lineaire deterministe / affine par morceaux
/ compositionnel binaire / booleen en reseau. C'est sur cet axe (la classe) que
l'on teste si le seuil d'interpretabilite glisse (relationnel) ou non (intrinseque).
"""
import numpy as np

from substrates import LogisticMap, RandomBooleanNetwork


# --------------------------------------------------------------------------- #
class Generator:
    name = "base"
    dim = 1
    binary = False
    kind = "base"

    def generate(self, steps, seed):
        raise NotImplementedError


class IID(Generator):
    """Bruit blanc gaussien. Aucune structure temporelle — plancher (et surrogate
    de toutes les autres classes)."""
    name = "iid"; kind = "aucune"

    def generate(self, steps, seed):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((steps, 1))


class Periodic(Generator):
    """Sinusoide de periode incommensurable avec la fenetre de plongement + bruit."""
    name = "periodic"; kind = "periodique"

    def __init__(self, period=17.0, noise=0.05):
        self.period = period; self.noise = noise

    def generate(self, steps, seed):
        rng = np.random.default_rng(seed)
        t = np.arange(steps)
        x = np.sin(2 * np.pi * t / self.period) + self.noise * rng.standard_normal(steps)
        return x.reshape(-1, 1)


class AR1(Generator):
    """Processus auto-regressif d'ordre 1 : x_t = phi x_{t-1} + bruit. Lineaire, memoire courte."""
    name = "ar1"; kind = "lineaire-court"

    def __init__(self, phi=0.9, noise=0.3):
        self.phi = phi; self.noise = noise

    def generate(self, steps, seed):
        rng = np.random.default_rng(seed)
        x = np.empty(steps); x[0] = rng.standard_normal()
        for t in range(1, steps):
            x[t] = self.phi * x[t - 1] + self.noise * rng.standard_normal()
        return x.reshape(-1, 1)


class RandomWalk(Generator):
    """Marche aleatoire : x_t = x_{t-1} + bruit. Processus integre — la PERSISTANCE
    est le predicteur optimal ; une regle lineaire globale a coef < 1 ou un MLP
    sur-ajuste DEGRADENT. Moteur d'inversion attendu (L1 > L3, L4)."""
    name = "randomwalk"; kind = "integre"

    def __init__(self, noise=1.0):
        self.noise = noise

    def generate(self, steps, seed):
        rng = np.random.default_rng(seed)
        return np.cumsum(self.noise * rng.standard_normal(steps)).reshape(-1, 1)


class AR3Osc(Generator):
    """AR(3) a racines complexes : quasi-periodique amorti. Lineaire, memoire longue —
    la regle lineaire est optimale ; la persistance echoue, le kNN paie sa variance."""
    name = "ar3_osc"; kind = "lineaire-long"

    def __init__(self, coefs=(0.6, -0.3, 0.45), noise=0.3):
        self.coefs = np.array(coefs); self.noise = noise

    def generate(self, steps, seed):
        rng = np.random.default_rng(seed)
        x = np.zeros(steps); x[:3] = rng.standard_normal(3)
        for t in range(3, steps):
            x[t] = self.coefs @ x[t - 3:t][::-1] + self.noise * rng.standard_normal()
        x = np.clip(x, -1e3, 1e3)                       # garde-fou de stabilite numerique
        return x.reshape(-1, 1)


class NoisyAR2(Generator):
    """AR(2) oscillant amorti + bruit d'observation. L'optimum est au MILIEU de
    l'echelle (L3 linear) : la persistance echoue (oscillation), le kNN paie sa
    variance sur signal bruite, le MLP ne fait que rattraper la regle lineaire
    (lineaire strictement inclus dans la classe MLP). Montre que l'optimum rode
    DANS LES DEUX SENS, pas seulement vers le plus simple."""
    name = "noisy_ar2"; kind = "oscillant-bruite"

    def __init__(self, c1=1.4, c2=-0.8, noise=0.35):
        self.c1 = c1; self.c2 = c2; self.noise = noise

    def generate(self, steps, seed):
        rng = np.random.default_rng(seed)
        x = np.zeros(steps); x[:2] = rng.standard_normal(2)
        for t in range(2, steps):
            x[t] = self.c1 * x[t - 1] + self.c2 * x[t - 2] + self.noise * rng.standard_normal()
        return np.clip(x, -50, 50).reshape(-1, 1)


class Logistic(Generator):
    """Carte logistique deterministe en regime chaotique (r=3.8). Le rapport un-pas
    est une parabole lisse : kNN/MLP la captent, la regle lineaire echoue (parabole
    != droite), la persistance echoue. Moteur d'inversion (L2,L4 > L3)."""
    name = "logistic"; kind = "non-lineaire-1D"

    def __init__(self, r=3.8):
        self._sub = LogisticMap(); self.r = r

    def generate(self, steps, seed):
        return self._sub.simulate(self.r, steps, seed)


class TAR(Generator):
    """Threshold AR (affine par morceaux) : pente differente selon le signe.
    Non-lineaire mais localement lineaire — teste kNN vs MLP, et la regle globale
    qui moyenne les deux regimes."""
    name = "tar"; kind = "affine-morceaux"

    def __init__(self, hi=0.9, lo=-0.4, noise=0.25):
        self.hi = hi; self.lo = lo; self.noise = noise

    def generate(self, steps, seed):
        rng = np.random.default_rng(seed)
        x = np.empty(steps); x[0] = rng.standard_normal()
        for t in range(1, steps):
            slope = self.hi if x[t - 1] > 0 else self.lo
            x[t] = slope * x[t - 1] + self.noise * rng.standard_normal()
        return x.reshape(-1, 1)


class XOR3(Generator):
    """Compositionnel binaire deterministe : x_t = x_{t-1} XOR x_{t-3}. Exige
    d'INTEGRER deux lags non adjacents (synergie Xi elevee). La regle lineaire reelle
    ne peut pas faire XOR (non separable) ; le MLP et le kNN a contexte exact gagnent.
    Moteur d'inversion fort (L4 >> L3)."""
    name = "xor3"; kind = "compositionnel-binaire"; binary = True

    def generate(self, steps, seed):
        rng = np.random.default_rng(seed)
        x = np.zeros(steps, dtype=int)
        x[:3] = (rng.random(3) < 0.5).astype(int)
        for t in range(3, steps):
            x[t] = x[t - 1] ^ x[t - 3]
        return x.reshape(-1, 1).astype(float)


class RBNGen(Generator):
    """Reseau booleen aleatoire de Kauffman au bord (K=2). Booleen en reseau :
    structure a 1 pas persistante (tables fixes). Reutilise substrates.RBN."""
    name = "rbn"; kind = "booleen-reseau"; binary = True

    def __init__(self, K=2, size=12):
        self._sub = RandomBooleanNetwork(size=size, ks=(K,)); self.K = K

    @property
    def dim(self):
        return self._sub.dim

    def generate(self, steps, seed):
        return self._sub.simulate(self.K, steps, seed).astype(float)


def all_generators():
    """Ordonnees par profondeur structurelle intuitive (axe de l'experience)."""
    return [IID(), Periodic(), AR1(), RandomWalk(), AR3Osc(), NoisyAR2(),
            Logistic(), TAR(), XOR3(), RBNGen()]
