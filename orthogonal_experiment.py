"""No Free Lunch plein — des inversions biais-fondamentales BIDIRECTIONNELLES ?

Panel d'interpretes a biais orthogonaux (non emboites) x classes de flux. On cherche
A,B,c1,c2 : A bat B sur c1 ET B bat A sur c2, les DEUX ecarts survivant au budget max
des deux interpretes -> aucun biais ne domine, jamais (forme forte de rho(X,M)).

Usage :
  python orthogonal_experiment.py --smoke
  python orthogonal_experiment.py            # -> results_orthogonal.json
"""
import argparse, json, time
import numpy as np

from generators import all_generators
from orthogonal_interpreters import make_panel
from predictor import make_windows

WINDOW = 4
MARG = 0.10            # marge d'inversion (plus stricte que l'echelle : on veut du net)
MAXB = {"fourier": 24, "local": 18}


def _split(traj, d, gap=None):
    X, Y = make_windows(traj, d)
    t = np.arange(d, d + len(X))                          # temps absolu de chaque cible
    gap = d if gap is None else gap
    n = len(X); ntr = int(n * 0.7)
    return (X[:ntr], Y[:ntr], t[:ntr]), (X[ntr + gap:], Y[ntr + gap:], t[ntr + gap:])


def _interp(pred, true):
    pred = np.nan_to_num(np.asarray(pred, float), nan=0.0, posinf=1e9, neginf=-1e9)
    sse = float(((pred - true) ** 2).sum()); sst = float(((true - true.mean(0)) ** 2).sum()) + 1e-12
    r = 1 - sse / sst
    return float(np.clip(r, 0, 1)) if np.isfinite(r) else 0.0


def matrix(classes, seeds, budgets=None, shuffle=False):
    panel0 = make_panel(1, WINDOW, False, budgets)
    names = [p.name for p in panel0]
    I, C, S = len(names), len(classes), len(seeds)
    acc = np.zeros((I, C, S))
    for gi, gen in enumerate(classes):
        for si, seed in enumerate(seeds):
            traj = gen.generate(900 if gen.dim > 4 else 1300, seed=seed * 17 + 3)
            if shuffle:
                rng = np.random.default_rng(seed * 91 + 2)
                traj = traj[rng.permutation(len(traj))]
            (Xtr, Ytr, ttr), (Xte, Yte, tte) = _split(traj, WINDOW)
            panel = make_panel(gen.dim, WINDOW, gen.binary, budgets)
            for ii, itp in enumerate(panel):
                itp.fit(Xtr, Ytr, ttr)
                acc[ii, gi, si] = _interp(itp.predict(Xte, tte), Yte)
    return names, acc.mean(2), acc.std(2) / np.sqrt(S)


def find_bidirectional(M, names, cls, Mmax=None, margin=MARG):
    """A bat B sur c1, B bat A sur c2 ; si Mmax fourni, exige la survie au budget max."""
    I, C = M.shape; found = []
    for a in range(I):
        for b in range(a + 1, I):
            wa = [c for c in range(C) if M[a, c] - M[b, c] > margin]    # classes ou a gagne
            wb = [c for c in range(C) if M[b, c] - M[a, c] > margin]    # ou b gagne
            if not (wa and wb):
                continue
            budget_ok = True
            if Mmax is not None:                                       # survie au budget max ?
                c1, c2 = wa[0], wb[0]
                budget_ok = (Mmax[a, c1] - Mmax[b, c1] > margin) and (Mmax[b, c2] - Mmax[a, c2] > margin)
            found.append(dict(A=names[a], B=names[b],
                              A_wins=[cls[c] for c in wa], B_wins=[cls[c] for c in wb],
                              budget_robust=bool(budget_ok)))
    return found


def find_cycle(M, names, cls, margin=MARG):
    """Cycle de domination A->B->C->A (A bat B sur une classe, etc.) = intransitivite."""
    I, C = M.shape
    beats = np.zeros((I, I), bool)
    for a in range(I):
        for b in range(I):
            if a != b and any(M[a, c] - M[b, c] > margin for c in range(C)):
                beats[a, b] = True
    cycles = []
    for a in range(I):
        for b in range(I):
            for c in range(I):
                if a < b and a < c and b != c and beats[a, b] and beats[b, c] and beats[c, a]:
                    cycles.append([names[a], names[b], names[c]])
    return cycles


def run(seeds, classes):
    cls = [g.name for g in classes]
    names, M, SEM = matrix(classes, seeds)
    _, Mmax, _ = matrix(classes, seeds, budgets=MAXB)
    _, Msur, _ = matrix(classes, seeds, shuffle=True)

    bidir = find_bidirectional(M, names, cls, Mmax=Mmax)
    bidir_robust = [b for b in bidir if b["budget_robust"]]
    bidir_sur = find_bidirectional(Msur, names, cls)
    cycles = find_cycle(M, names, cls)

    iid = cls.index("iid") if "iid" in cls else None
    H1 = bool(M[:, iid].max() < 0.05) if iid is not None else None
    # H2 : aucun interprete dans 0.1 du max-colonne sur TOUTES les classes structurees
    struct = [c for c in range(len(cls)) if cls[c] != "iid"]
    colmax = M.max(0)
    universal = [names[i] for i in range(len(names))
                 if all(M[i, c] >= colmax[c] - MARG for c in struct)]
    H2 = (len(universal) == 0)
    H3 = len(bidir_robust) >= 1
    H4 = (len(cycles) >= 1) or (len(bidir) >= 2)
    H5 = (Msur.max() < 0.05 + 0.21) and (len(bidir_sur) == 0)   # rbn statique tolere ; 0 inversion

    if H1 and H3 and (len(bidir_sur) == 0):
        verdict = "NO FREE LUNCH PLEIN (bidirectionnel)"
    elif H1 and H2 and not H3:
        verdict = "MONO-DIRECTIONNEL"
    else:
        verdict = "UN BIAIS DOMINE / autre"

    return dict(interpreters=names, biases=[p.bias for p in make_panel(1, WINDOW, False)],
                classes=cls, kinds=[g.kind for g in classes],
                interp=M.tolist(), interp_max=Mmax.tolist(), interp_surrogate=Msur.tolist(),
                interp_sem=SEM.tolist(),
                bidirectional=bidir, bidirectional_robust=bidir_robust,
                bidirectional_surrogate=bidir_sur, cycles=cycles, universal=universal,
                H1=H1, H2=bool(H2), H3=bool(H3), H4=bool(H4), H5=bool(H5), verdict=verdict)


def report(res):
    names, cls, M = res["interpreters"], res["classes"], np.array(res["interp"])
    print("\n" + "=" * 80)
    print("MATRICE interp (R^2 un-pas held-out)   lignes=biais orthogonaux  colonnes=classes")
    print("=" * 80)
    print("interprete".ljust(12) + " ".join(c[:7].rjust(7) for c in cls))
    for i, n in enumerate(names):
        print(f"{n[:11].ljust(12)}" + " ".join(f"{M[i, j]:.2f}".rjust(7) for j in range(len(cls))))
    print("\nfoyer de chaque classe (argmax) :")
    for j, c in enumerate(cls):
        if M[:, j].max() >= 0.05:
            print(f"  {c:<12} -> {names[int(M[:, j].argmax())]}  ({M[:, j].max():.2f})")
    print("\nINVERSIONS BIDIRECTIONNELLES (A bat B ici, B bat A là) :")
    for b in res["bidirectional"]:
        tag = "BUDGET-ROBUSTE" if b["budget_robust"] else "budget-sensible"
        print(f"  {b['A']} ⇄ {b['B']}   {b['A']}+:{b['A_wins']}  {b['B']}+:{b['B_wins']}   [{tag}]")
    print(f"\n  cycles de domination (intransitivité) : {res['cycles']}")
    print(f"  interprète universel (gagne partout) : {res['universal'] or 'AUCUN'}")
    print(f"  inversions bidir sur surrogate (doit=0) : {len(res['bidirectional_surrogate'])}")
    print("\nVERDICTS :")
    for h in ("H1", "H2", "H3", "H4", "H5"):
        print(f"  {h} = {res[h]}")
    print("\n" + "#" * 80 + f"\nVERDICT : {res['verdict']}\n" + "#" * 80)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results_orthogonal.json"); a = ap.parse_args()
    classes = all_generators()
    if a.smoke:
        classes = [c for c in classes if c.name in ("iid", "periodic", "randomwalk", "logistic", "xor3")]
        seeds = [0, 1]
    else:
        seeds = list(range(5))
    t0 = time.time()
    print(f"[orthogonal] {len(classes)} classes x {len(make_panel(1,WINDOW,False))} biais x {len(seeds)} seeds ...", flush=True)
    res = run(seeds, classes); res["seconds"] = round(time.time() - t0, 1)
    report(res)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nresultats -> {a.out}  ({res['seconds']}s)")


if __name__ == "__main__":
    main()
