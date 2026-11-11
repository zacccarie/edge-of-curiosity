"""Experience decisive — parametrage minimal de l'interpretation : intrinseque ou relationnel ?

Pour chaque classe de flux x chaque barreau de l'echelle x chaque seed :
  - interp = R^2 un-pas held-out (capture de structure)
  - triplet (Gamma, Xi, beta)
Puis : seuil d'apparition L*(classe) ; test d'INVERSION (un barreau plus bas bat un
plus haut -> ordre non total -> relationnel) ; controle surrogate. Verdicts pre-enregistres.

Sortie : results_ladder.json + resume ASCII (numpy uniquement).

Usage :
  python ladder_experiment.py --smoke    # rapide (~20s) pour valider
  python ladder_experiment.py            # run complet -> results_ladder.json
"""
import argparse
import json
import time
import numpy as np

from generators import all_generators
from interpreters import make_ladder
from predictor import make_windows
from metrics import (interp_score, beta_residual, xi_synergy_captured, xi_data,
                     gamma_organization)

WINDOW = 4
MARGE = 0.05
H_ROLL = 60


def _split(traj, d, train_frac=0.7, gap=None):
    """Fenetres causales, split temporel avec un trou (anti-fuite)."""
    X, Y = make_windows(traj, d)
    gap = d if gap is None else gap
    n = len(X)
    ntr = int(n * train_frac)
    Xtr, Ytr = X[:ntr], Y[:ntr]
    Xte, Yte = X[ntr + gap:], Y[ntr + gap:]
    return Xtr, Ytr, Xte, Yte


def _context_lags(Xte, dim, d):
    """Les d lags isoles (dim 0) alignes avec la cible, pour Xi synergique."""
    return [Xte[:, (d - k) * dim + 0] for k in range(1, d + 1)]


def eval_cell(gen, interp_factory_idx, seed, with_triplet=True, shuffle=False):
    """Un (classe, barreau, seed). Renvoie dict interp + triplet."""
    dim = gen.dim
    T = 900 if dim > 4 else 1300
    traj = gen.generate(T, seed=seed * 17 + 3)
    if shuffle:
        rng = np.random.default_rng(seed * 91 + 2)
        traj = traj[rng.permutation(len(traj))]
    Xtr, Ytr, Xte, Yte = _split(traj, WINDOW)
    ladder = make_ladder(dim, WINDOW, gen.binary, seed=seed)
    itp = ladder[interp_factory_idx]
    itp.fit(Xtr, Ytr)
    pred = itp.predict_onestep(Xte)
    interp = interp_score(pred, Yte)
    out = dict(level=itp.level, name=itp.name, interp=interp)
    if not with_triplet:
        return out
    resid = Yte - pred
    out["beta"] = beta_residual(resid)
    out["xi"] = xi_synergy_captured(pred, Yte, _context_lags(Xte, dim, WINDOW))
    # Gamma : rollout libre sur trajectoire held-out fraiche
    tr2 = gen.generate(WINDOW + H_ROLL + 2, seed=seed * 53 + 901)
    win = tr2[:WINDOW].reshape(-1).astype(float)
    roll = np.empty((H_ROLL, dim))
    for h in range(H_ROLL):
        nxt = itp.step(win)
        roll[h] = nxt
        win = np.concatenate([win[dim:], np.asarray(nxt, float).ravel()])
    out["gamma"] = gamma_organization(roll, tr2[WINDOW:WINDOW + H_ROLL])
    return out


def run(seeds, classes, n_levels=6, do_surrogate=True):
    G = len(classes); L = n_levels; S = len(seeds)
    interp = np.zeros((L, G, S))
    gamma = np.zeros((L, G, S)); xi = np.zeros((L, G, S)); beta = np.zeros((L, G, S))
    interp_sur = np.zeros((L, G, S))
    xi_dat = np.zeros(G)
    names = [None] * L
    class_names = [g.name for g in classes]
    class_kinds = [g.kind for g in classes]

    for gi, gen in enumerate(classes):
        long_traj = gen.generate(2200 if gen.dim <= 4 else 1400, seed=12345)
        xi_dat[gi] = xi_data(long_traj)
        for si, seed in enumerate(seeds):
            for li in range(L):
                cell = eval_cell(gen, li, seed, with_triplet=True)
                names[li] = cell["name"]
                interp[li, gi, si] = cell["interp"]
                gamma[li, gi, si] = cell["gamma"]
                xi[li, gi, si] = cell["xi"]
                beta[li, gi, si] = cell["beta"]
                if do_surrogate:
                    interp_sur[li, gi, si] = eval_cell(gen, li, seed, with_triplet=False,
                                                       shuffle=True)["interp"]

    M = interp.mean(2); SEM = interp.std(2) / np.sqrt(S)
    Msur = interp_sur.mean(2); SEMsur = interp_sur.std(2) / np.sqrt(S)
    Mtemp = np.clip(M - Msur, 0.0, 1.0)                   # structure TEMPORELLE (surrogate paire)
    SEMtemp = np.sqrt(SEM ** 2 + SEMsur ** 2)
    Gm, Xm, Bm = gamma.mean(2), xi.mean(2), beta.mean(2)
    stab = 1.0 - interp.std(2) / (M + 1e-9)               # stabilite inter-seed (diagnostic)

    # --- seuil d'apparition L*(classe) ---
    onset = []
    onset_strong = []                                    # exploratoire : interp > 0.5
    for gi in range(G):
        hits = np.where(M[:, gi] > MARGE)[0]
        onset.append(int(hits.min()) if len(hits) else None)
        hs = np.where(M[:, gi] > 0.5)[0]
        onset_strong.append(int(hs.min()) if len(hs) else None)

    # --- inversions (decisif) et crossovers ---
    inversions = _find_inversions(M, SEM, class_names, names)
    inversions_temporal = _find_inversions(Mtemp, SEMtemp, class_names, names)   # apres surrogate paire
    inversions_sur = _find_inversions(Msur, SEMsur, class_names, names)
    crossovers = _find_crossovers(M, class_names)
    sur_imax = int(np.argmax(Msur))
    sur_cell = (names[sur_imax // G], class_names[sur_imax % G], round(float(Msur.max()), 3))

    # --- onset Xi (apparition de l'integration) ---
    xi_onset = []
    for gi in range(G):
        hits = np.where(Xm[:, gi] > 0.02)[0]
        xi_onset.append(int(hits.min()) if len(hits) else None)

    # --- verdicts pre-enregistres ---
    iid_idx = class_names.index("iid") if "iid" in class_names else None
    H1 = (M[:, iid_idx].max() < MARGE) if iid_idx is not None else None
    struct_onsets = [o for gi, o in enumerate(onset) if class_names[gi] != "iid" and o is not None]
    H2 = (max(struct_onsets) - min(struct_onsets) >= 2) if struct_onsets else False
    H3 = len(inversions) >= 1
    H4 = len(crossovers) >= 1
    xi_on_struct = [o for gi, o in enumerate(xi_onset) if class_names[gi] != "iid" and o is not None]
    H5 = (max(xi_on_struct) - min(xi_on_struct) >= 2) if xi_on_struct else False
    # H6 : controle surrogate. H6a (clause naive) = aucune structure ne survit au melange ;
    # tripee par la structure STATIQUE du rbn (noeuds geles). H6b (clause decisive) =
    # aucune inversion ne survit au melange. H6c = les inversions tiennent sur la
    # structure TEMPORELLE (real - surrogate, surrogate paire).
    H6a = bool(Msur.max() < MARGE) if do_surrogate else None
    H6b = (len(inversions_sur) == 0) if do_surrogate else None
    H6c = (len(inversions_temporal) >= 1) if do_surrogate else None
    H6 = bool(H6b and H6c) if do_surrogate else None

    if H1 and H3 and (H6 if do_surrogate else True):
        verdict = "RELATIONNEL (bord)"
    elif H1 and not H3 and not H4:
        verdict = "INTRINSEQUE (mainstream)"
    else:
        verdict = "MIXTE / frontiere"

    return dict(
        levels=names, classes=class_names, kinds=class_kinds, seeds=list(seeds),
        interp=M.tolist(), interp_sem=SEM.tolist(), interp_surrogate=Msur.tolist(),
        gamma=Gm.tolist(), xi=Xm.tolist(), beta=Bm.tolist(),
        interp_temporal=Mtemp.tolist(),
        xi_data=xi_dat.tolist(), stability=stab.tolist(),
        onset=onset, onset_strong=onset_strong, xi_onset=xi_onset,
        inversions=inversions, inversions_temporal=inversions_temporal,
        inversions_surrogate=inversions_sur, crossovers=crossovers, surrogate_max_cell=sur_cell,
        H1=bool(H1) if H1 is not None else None, H2=bool(H2), H3=bool(H3),
        H4=bool(H4), H5=bool(H5),
        H6=(bool(H6) if H6 is not None else None),
        H6a=H6a, H6b=H6b, H6c=H6c,
        verdict=verdict,
    )


def _find_inversions(M, SEM, class_names, level_names, margin=MARGE):
    """Un barreau plus bas (a<b) bat un plus haut sur une classe, au-dela du bruit."""
    L, G = M.shape
    found = []
    for c in range(G):
        for a in range(L):
            for b in range(a + 1, L):
                gap = M[a, c] - M[b, c]
                sem = np.sqrt(SEM[a, c] ** 2 + SEM[b, c] ** 2)
                if gap > margin and gap > 2 * sem:
                    found.append(dict(klass=class_names[c], low=level_names[a],
                                      high=level_names[b], low_lvl=a, high_lvl=b,
                                      interp_low=round(float(M[a, c]), 3),
                                      interp_high=round(float(M[b, c]), 3),
                                      gap=round(float(gap), 3)))
    return found


def _find_crossovers(M, class_names, margin=MARGE):
    """Paires (a,b) mutuellement non-dominantes : a gagne sur une classe, b sur une autre."""
    L, G = M.shape
    found = []
    for a in range(L):
        for b in range(a + 1, L):
            a_wins = [class_names[c] for c in range(G) if M[a, c] - M[b, c] > margin]
            b_wins = [class_names[c] for c in range(G) if M[b, c] - M[a, c] > margin]
            if a_wins and b_wins:
                found.append(dict(a=a, b=b, a_wins_on=a_wins, b_wins_on=b_wins))
    return found


# --------------------------------------------------------------------------- #
def _barcell(v, width=6):
    n = int(round(width * np.clip(v, 0, 1)))
    return "#" * n + "·" * (width - n)


def print_report(res):
    L = res["levels"]; C = res["classes"]; M = np.array(res["interp"])
    print("\n" + "=" * 78)
    print("MATRICE interp (R^2 un-pas, held-out)   lignes=barreaux  colonnes=classes")
    print("=" * 78)
    head = "barreau".ljust(13) + " ".join(c[:7].rjust(7) for c in C)
    print(head)
    for li, lname in enumerate(L):
        row = f"L{li} {lname[:9]}".ljust(13)
        row += " ".join(f"{M[li, gi]:.2f}".rjust(7) for gi in range(len(C)))
        print(row)
    print("\nseuil d'apparition L*(classe)  (1er barreau avec interp > marge) :")
    for gi, c in enumerate(C):
        o = res["onset"][gi]
        print(f"  {c:<22} L* = {'∞' if o is None else o}   ({res['kinds'][gi]})")
    print(f"\n  -> etendue de L* sur classes structurees : H2(glissement)={res['H2']}")
    print("\nINVERSIONS (barreau bas bat barreau haut — signature relationnelle) :")
    if res["inversions"]:
        for inv in res["inversions"]:
            print(f"  • sur {inv['klass']:<20} {inv['low']}(L{inv['low_lvl']})={inv['interp_low']}"
                  f"  >  {inv['high']}(L{inv['high_lvl']})={inv['interp_high']}   (Δ={inv['gap']})")
    else:
        print("  (aucune — ordre total -> intrinseque)")
    print(f"\n  crossovers (paires non-dominantes) : {len(res['crossovers'])}")
    print(f"  inversions sur structure temporelle (real-surrogate) : {len(res['inversions_temporal'])}")
    print(f"  inversions sur surrogate (doit etre 0) : {len(res['inversions_surrogate'])}")
    sc = res["surrogate_max_cell"]
    print(f"  max interp surrogate : {sc[2]} ({sc[0]}/{sc[1]} — structure statique, cf. rbn gele)")
    print("\nVERDICTS pre-enregistres :")
    print(f"  H1 plancher iid muet      : {res['H1']}")
    print(f"  H2 L* glisse (>=2)        : {res['H2']}")
    print(f"  H3 inversion robuste      : {res['H3']}   <<< decisif")
    print(f"  H4 crossover              : {res['H4']}")
    print(f"  H5 onset Xi glisse        : {res['H5']}")
    print(f"  H6 surrogate (H6b∧H6c)    : {res['H6']}   [H6a naive={res['H6a']} H6b 0-inv-sur={res['H6b']} H6c inv-temporelles={res['H6c']}]")
    print("\n" + "#" * 78)
    print(f"VERDICT : {res['verdict']}")
    print("#" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results_ladder.json")
    args = ap.parse_args()

    classes = all_generators()
    if args.smoke:
        classes = [c for c in classes if c.name in ("iid", "randomwalk", "logistic", "xor3")]
        seeds = [0, 1]
        do_sur = True
    else:
        seeds = list(range(5))
        do_sur = True

    t0 = time.time()
    print(f"[ladder] {len(classes)} classes x 6 barreaux x {len(seeds)} seeds ...", flush=True)
    res = run(seeds, classes, do_surrogate=do_sur)
    res["seconds"] = round(time.time() - t0, 1)
    print_report(res)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nresultats -> {args.out}   ({res['seconds']}s)")


if __name__ == "__main__":
    main()
