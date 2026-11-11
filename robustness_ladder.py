"""Robustesse des inversions — separer les inversions FONDAMENTALES (biais inductif,
qu'aucun hyperparametre ne corrige) des inversions de BUDGET (le MLP sous-entraine).

Arms :
  budget  : MLP hid/budget tres augmentes vs kNN sur {logistic, xor3, randomwalk}
  knn_k   : sensibilite de kNN a k
  window  : sensibilite a la fenetre de plongement D
  seeds   : plus de seeds pour resserrer la SEM des inversions coeur

Usage : python robustness_ladder.py --arm budget --seeds 8
"""
import argparse
import numpy as np

import interpreters as I
from interpreters import Marginal, Persistence, KNN, Linear, MLPInterp
from generators import RandomWalk, Logistic, XOR3, AR1, AR3Osc, TAR, Periodic
from predictor import make_windows
from metrics import interp_score

WINDOW = 4


def _eval(gen, itp, seed, window=WINDOW):
    dim = gen.dim
    T = 1300
    traj = gen.generate(T, seed=seed * 17 + 3)
    X, Y = make_windows(traj, window)
    n = len(X); ntr = int(n * 0.7)
    itp.fit(X[:ntr], Y[:ntr])
    pred = itp.predict_onestep(X[ntr + window:])
    return interp_score(pred, Y[ntr + window:])


def _mean(gen, factory, seeds, window=WINDOW):
    vals = [_eval(gen, factory(s), s, window) for s in seeds]
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(seeds)))


def arm_budget(seeds):
    print("\n=== ARM budget : le MLP a haute capacite rattrape-t-il kNN / la persistance ? ===")
    classes = {"randomwalk": RandomWalk(), "logistic": Logistic(), "xor3": XOR3()}
    configs = [
        ("mlp hid=8  b=1500",  lambda s, g: MLPInterp(g.dim, WINDOW, g.binary, hid=8,  budget=1500,  seed=s)),
        ("mlp hid=32 b=2500",  lambda s, g: MLPInterp(g.dim, WINDOW, g.binary, hid=32, budget=2500,  seed=s)),
        ("mlp hid=64 b=20000", lambda s, g: MLPInterp(g.dim, WINDOW, g.binary, hid=64, budget=20000, seed=s)),
        ("mlp hid=128 b=60000",lambda s, g: MLPInterp(g.dim, WINDOW, g.binary, hid=128,budget=60000, seed=s)),
    ]
    refs = {
        "randomwalk": ("persistence", lambda s, g: Persistence(g.dim, WINDOW, g.binary)),
        "logistic":   ("knn k=5",     lambda s, g: KNN(g.dim, WINDOW, g.binary, k=5)),
        "xor3":       ("knn k=5",     lambda s, g: KNN(g.dim, WINDOW, g.binary, k=5)),
    }
    for cname, gen in classes.items():
        rname, rfac = refs[cname]
        rm, rs = _mean(gen, lambda s: rfac(s, gen), seeds)
        print(f"\n  [{cname}]  reference {rname} = {rm:.3f} ± {rs:.3f}")
        for label, fac in configs:
            m, sm = _mean(gen, lambda s: fac(s, gen), seeds)
            verdict = "INVERSION TIENT" if rm > m + 0.05 else "rattrape (budget)"
            print(f"    {label:<22} interp = {m:.3f} ± {sm:.3f}   ->  {verdict}  (ref-mlp = {rm - m:+.3f})")


def arm_knn_k(seeds):
    print("\n=== ARM knn_k : la victoire de kNN depend-elle de k ? ===")
    for cname, gen in (("logistic", Logistic()), ("xor3", XOR3()), ("randomwalk", RandomWalk())):
        print(f"\n  [{cname}]")
        for k in (1, 3, 5, 15, 50):
            m, sm = _mean(gen, lambda s: KNN(gen.dim, WINDOW, gen.binary, k=k), seeds)
            print(f"    knn k={k:<3} interp = {m:.3f} ± {sm:.3f}")


def arm_window(seeds):
    print("\n=== ARM window : structure d'inversion robuste a la fenetre D ? ===")
    for D in (2, 3, 6):
        print(f"\n  D={D}")
        for cname, gen in (("randomwalk", RandomWalk()), ("logistic", Logistic()), ("xor3", XOR3())):
            facs = [("persist", lambda s: Persistence(gen.dim, D, gen.binary)),
                    ("knn", lambda s: KNN(gen.dim, D, gen.binary, k=5)),
                    ("linear", lambda s: Linear(gen.dim, D, gen.binary)),
                    ("mlp32", lambda s: MLPInterp(gen.dim, D, gen.binary, hid=32, budget=2500, seed=s))]
            vals = {name: _mean(gen, fac, seeds, window=D)[0] for name, fac in facs}
            print(f"    {cname:<11} " + "  ".join(f"{n}={v:.2f}" for n, v in vals.items()))


def arm_seeds(seeds):
    print(f"\n=== ARM seeds : inversions coeur sur {len(seeds)} seeds (SEM resserree) ===")
    tests = [
        ("randomwalk persist>knn", RandomWalk(),
         lambda s, g: Persistence(g.dim, WINDOW, g.binary), lambda s, g: KNN(g.dim, WINDOW, g.binary, 5)),
        ("randomwalk persist>mlp", RandomWalk(),
         lambda s, g: Persistence(g.dim, WINDOW, g.binary), lambda s, g: MLPInterp(g.dim, WINDOW, g.binary, hid=32, budget=2500, seed=s)),
        ("logistic knn>linear", Logistic(),
         lambda s, g: KNN(g.dim, WINDOW, g.binary, 5), lambda s, g: Linear(g.dim, WINDOW, g.binary)),
        ("xor3 knn>linear", XOR3(),
         lambda s, g: KNN(g.dim, WINDOW, g.binary, 5), lambda s, g: Linear(g.dim, WINDOW, g.binary)),
    ]
    for label, gen, lowf, highf in tests:
        lm, ls = _mean(gen, lambda s: lowf(s, gen), seeds)
        hm, hs = _mean(gen, lambda s: highf(s, gen), seeds)
        gap = lm - hm; sem = np.sqrt(ls ** 2 + hs ** 2)
        ok = gap > 0.05 and gap > 2 * sem
        print(f"  {label:<26} low={lm:.3f}±{ls:.3f}  high={hm:.3f}±{hs:.3f}  Δ={gap:+.3f}  "
              f"({gap/ (sem+1e-9):.1f}σ)  {'ROBUSTE' if ok else 'fragile'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["budget", "knn_k", "window", "seeds", "all"], default="budget")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    arms = [args.arm] if args.arm != "all" else ["budget", "knn_k", "window", "seeds"]
    for a in arms:
        {"budget": arm_budget, "knn_k": arm_knn_k, "window": arm_window, "seeds": arm_seeds}[a](seeds)
