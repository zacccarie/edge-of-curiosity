"""Mesures : interp (capture de structure) + triplet (Gamma, Xi, beta).

Instanciation operationnelle assumee (Xi reste un *template*, cf. xi-observable) :
  interp = R^2 un-pas held-out (fraction de variance captee, clip [0,1])
  beta   = pente spectrale (log-log) du RESIDU un-pas  (residu blanc -> beta~0)
  Xi     = information predictive SYNERGIQUE : MI(pred ; futur) au-dela du meilleur
           lag isole  (marque l'apparition de l'integration, irreductible aux lookups)
  Gamma  = organisation du rollout libre : 1 - distance ACF(rollout, verite)
"""
import numpy as np


# --------------------------------------------------------------------------- #
def interp_score(pred, true):
    """R^2 held-out poole sur les dimensions, clip [0,1]. 0 = pas mieux que la
    marginale ; 1 = structure entierement captee."""
    pred = np.nan_to_num(np.asarray(pred, float), nan=0.0, posinf=1e9, neginf=-1e9)
    true = np.asarray(true, float)
    sse = float(((pred - true) ** 2).sum())
    sst = float(((true - true.mean(0)) ** 2).sum()) + 1e-12
    r2 = 1.0 - sse / sst
    return float(np.clip(r2, 0.0, 1.0)) if np.isfinite(r2) else 0.0


# --------------------------------------------------------------------------- #
def _periodogram(x, seg=128, overlap=0.5):
    """Welch simplifie (fenetre de Hann, segments moyennes)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if n < seg:
        seg = n
    step = max(1, int(seg * (1 - overlap)))
    win = np.hanning(seg)
    psum = None; cnt = 0
    for start in range(0, n - seg + 1, step):
        s = x[start:start + seg] * win
        p = np.abs(np.fft.rfft(s)) ** 2
        psum = p if psum is None else psum + p
        cnt += 1
    if cnt == 0:
        s = (x[:seg] if n >= seg else np.pad(x, (0, seg - n))) * win
        psum = np.abs(np.fft.rfft(s)) ** 2; cnt = 1
    freqs = np.fft.rfftfreq(seg)
    return freqs, psum / cnt


def beta_residual(residual, lo=0.06, hi=0.45):
    """Pente spectrale du residu sur une bande mediane (log-log). Residu blanchi
    -> pente ~ 0 (structure captee). Multi-dim : moyenne des pentes par dim."""
    r = np.asarray(residual, float)
    if r.ndim == 1:
        r = r[:, None]
    slopes = []
    for j in range(r.shape[1]):
        rj = r[:, j]
        if np.var(rj) < 1e-12:
            continue                                     # residu nul -> structure pleine
        f, p = _periodogram(rj)
        band = (f >= lo) & (f <= hi) & (p > 0)
        if band.sum() < 4:
            continue
        lf, lp = np.log10(f[band]), np.log10(p[band])
        A = np.vstack([lf, np.ones_like(lf)]).T
        slope = float(np.linalg.lstsq(A, lp, rcond=None)[0][0])
        slopes.append(slope)
    return float(np.mean(slopes)) if slopes else 0.0


# --------------------------------------------------------------------------- #
def _bin(x, B):
    """Discretisation par quantiles (B casiers). Binaire -> 2 casiers."""
    x = np.asarray(x, float).ravel()
    u = np.unique(x)
    if len(u) <= 2:
        return (x > (u.min() + u.max()) / 2).astype(int) if len(u) == 2 else np.zeros(len(x), int)
    q = np.quantile(x, np.linspace(0, 1, B + 1)[1:-1])
    return np.digitize(x, q)


def mi_discrete(a, b):
    """Information mutuelle (nats) entre deux etiquetages discrets."""
    a = np.asarray(a, int); b = np.asarray(b, int)
    n = len(a)
    A = a.max() + 1; Bn = b.max() + 1
    joint = np.zeros((A, Bn))
    np.add.at(joint, (a, b), 1.0)
    joint /= n
    pa = joint.sum(1, keepdims=True); pb = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())


def xi_synergy_captured(pred, true, context_lags, B=5):
    """Xi capte = MI(pred ; futur) - max_k MI(lag_k ; futur), clip a 0.
    > 0 ssi le modele exploite l'information JOINTE au-dela de tout lag isole
    (integration). context_lags : liste de colonnes (chaque lag), meme longueur que true."""
    yb = _bin(true[:, 0] if true.ndim > 1 else true, B)
    pb = _bin(pred[:, 0] if pred.ndim > 1 else pred, B)
    mi_pred = mi_discrete(pb, yb)
    mi_single = [mi_discrete(_bin(lag, B), yb) for lag in context_lags]
    best_single = max(mi_single) if mi_single else 0.0
    return float(max(0.0, mi_pred - best_single))


def xi_data(traj, lags=(1, 2, 3), B=4):
    """Synergie INTRINSEQUE de la classe : MI(x_t ; contexte joint) - Sum MI(x_t ; lag_k).
    Propriete de la donnee (combien d'integration la classe exige)."""
    x = traj[:, 0] if traj.ndim > 1 else np.asarray(traj)
    L = max(lags)
    y = x[L:]
    cols = [x[L - k:len(x) - k] for k in lags]
    yb = _bin(y, B)
    binned = [_bin(c, B) for c in cols]
    # symbole joint du contexte
    joint = np.zeros(len(yb), int)
    for c in binned:
        joint = joint * B + c
    _, joint = np.unique(joint, return_inverse=True)
    mi_joint = mi_discrete(joint, yb)
    mi_sum = sum(mi_discrete(c, yb) for c in binned)
    return float(mi_joint - mi_sum)                      # > 0 : synergie ; < 0 : redondance


# --------------------------------------------------------------------------- #
def _acf(x, L):
    x = np.asarray(x, float) - np.mean(x)
    v = np.dot(x, x) + 1e-12
    return np.array([np.dot(x[:len(x) - k], x[k:]) / v for k in range(1, L + 1)])


def gamma_organization(rollout, truth, L=20):
    """Organisation = 1 - distance ACF entre rollout libre et verite. Effondrement
    (point fixe, var~0) ou divergence -> Gamma bas ; attracteur reproduit -> haut."""
    r = np.asarray(rollout, float)
    t = np.asarray(truth, float)
    r = r[:, 0] if r.ndim > 1 else r
    t = t[:, 0] if t.ndim > 1 else t
    r = np.clip(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0), -50, 50)
    if np.var(r) < 1e-9 or np.var(t) < 1e-9:
        return 0.0
    Lf = min(L, len(r) - 1, len(t) - 1)
    if Lf < 2:
        return 0.0
    d = np.mean(np.abs(_acf(r, Lf) - _acf(t, Lf)))
    return float(np.clip(1.0 - d, 0.0, 1.0))
