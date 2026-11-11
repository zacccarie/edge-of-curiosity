"""Orchestration de l'experience decisive + verdicts pre-enregistres.

Pour chaque substrat : localiser le bord (independamment du predicteur),
mesurer LP(theta), lancer 3 agents, appliquer les criteres de PREREGISTRATION.md.
Sortie : results.json + resume ASCII (numpy only, pas de matplotlib).

Usage :
  python experiment.py            # run complet (3 substrats, 5 seeds)
  python experiment.py --smoke    # run rapide (logistique, 1 seed) pour valider
"""
import argparse
import json
import time
import numpy as np

from substrates import LogisticMap, CoupledMapLattice, RandomBooleanNetwork
from predictor import intrinsic_lp
from curiosity import run_agent


def _rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def entropy(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def bar(v, vmax, width=28):
    n = int(round(width * v / (vmax + 1e-12))) if vmax > 0 else 0
    n = max(0, min(width, n))                          # clamp (LP peut etre negatif)
    return "#" * n + "-" * (width - n)


def run_substrate(sub, seeds, lp_kw, agent_kw, do_surrogate=True):
    thetas = sub.thetas
    A = len(thetas)
    S = len(seeds)
    crit = np.zeros((S, A))
    lp = np.zeros((S, A))
    lp_sur = np.zeros((S, A))
    d_lp = np.zeros((S, A))
    d_err = np.zeros((S, A))
    d_rnd = np.zeros((S, A))

    for si, seed in enumerate(seeds):
        for ai, th in enumerate(thetas):
            crit[si, ai] = sub.criticality_raw(th, seed)
            lp[si, ai] = intrinsic_lp(sub, th, seed, **lp_kw)[0]
            if do_surrogate:
                lp_sur[si, ai] = intrinsic_lp(sub, th, seed, shuffle=True, **lp_kw)[0]
        d_lp[si] = run_agent(sub, "lp", seed, **agent_kw)
        d_err[si] = run_agent(sub, "error", seed, **agent_kw)
        d_rnd[si] = run_agent(sub, "random", seed, **agent_kw)

    crit_m = crit.mean(0)
    lp_m = lp.mean(0)
    lp_sur_m = lp_sur.mean(0)
    d_lp_m, d_err_m, d_rnd_m = d_lp.mean(0), d_err.mean(0), d_rnd.mean(0)

    theta_c = sub.locate_edge(thetas, crit_m)
    peaked = sub.peaked_from_raw(crit_m)
    binw = float(np.mean(np.diff(thetas)))
    tol = 1.0 if sub.name == "rbn" else 1.5 * binw

    argmax_lp = float(thetas[int(np.argmax(lp_m))])
    argmax_dlp = float(thetas[int(np.argmax(d_lp_m))])
    argmax_derr = float(thetas[int(np.argmax(d_err_m))])
    argmax_lp_sur = float(thetas[int(np.argmax(lp_sur_m))]) if do_surrogate else None

    # --- verdicts pre-enregistres ---
    H1 = abs(argmax_lp - theta_c) <= tol
    unif_H = np.log(A)
    H2 = (abs(argmax_dlp - theta_c) <= tol) and (entropy(d_lp_m) < 0.9 * unif_H)
    chaos_ok = (argmax_derr > theta_c) if sub.chaos_side == "high" else (argmax_derr < theta_c)
    H3 = chaos_ok and (abs(argmax_derr - argmax_dlp) > 0.5 * binw if sub.name != "rbn"
                       else argmax_derr != argmax_dlp)
    H4 = float(d_rnd_m.max()) < 1.5 / A
    sp = spearman(lp_m, peaked)
    H5 = sp >= 0.5
    H6_pass = H1 and H3
    surrogate_clean = (abs(argmax_lp_sur - theta_c) > tol) if do_surrogate else None

    return dict(
        name=sub.name, theta_label=sub.theta_label, chaos_side=sub.chaos_side,
        thetas=[float(t) for t in thetas], theta_c=theta_c, tol=tol, binw=binw,
        crit=crit_m.tolist(), peaked=peaked.tolist(),
        lp=lp_m.tolist(), lp_std=lp.std(0).tolist(), lp_surrogate=lp_sur_m.tolist(),
        dwell_lp=d_lp_m.tolist(), dwell_error=d_err_m.tolist(), dwell_random=d_rnd_m.tolist(),
        argmax_lp=argmax_lp, argmax_dwell_lp=argmax_dlp, argmax_dwell_error=argmax_derr,
        argmax_lp_surrogate=argmax_lp_sur, spearman=sp,
        H1=bool(H1), H2=bool(H2), H3=bool(H3), H4=bool(H4), H5=bool(H5),
        H6_pass=bool(H6_pass), surrogate_clean=(bool(surrogate_clean) if do_surrogate else None),
    )


def print_report(res):
    thetas = np.array(res["thetas"])
    lab = res["theta_label"]
    print("\n" + "=" * 70)
    print(f"SUBSTRAT : {res['name'].upper()}   bouton={lab}   bord theta_c={res['theta_c']:.3f}"
          f"   (cote chaos={res['chaos_side']})")
    print("=" * 70)
    lp = np.array(res["lp"])
    dlp = np.array(res["dwell_lp"])
    derr = np.array(res["dwell_error"])
    crit = np.array(res["peaked"])
    lpmax, dlpmax, derrmax = lp.max(), dlp.max(), derr.max()
    cmin, cmax = crit.min(), crit.max()
    print(f"{lab:>7} | {'criticalite(piquee)':^22} | {'LP intrinseque':^22} | "
          f"{'dwell curiosite':^22} | {'dwell erreur':^22}")
    for i, t in enumerate(thetas):
        cnorm = (crit[i] - cmin) / (cmax - cmin + 1e-12)
        mark = " <-- bord" if abs(t - res["theta_c"]) <= res["binw"] / 1.5 else ""
        print(f"{t:7.3f} | {bar(cnorm,1,22)} | {bar(lp[i],lpmax,22)} | "
              f"{bar(dlp[i],dlpmax,22)} | {bar(derr[i],derrmax,22)}{mark}")
    print(f"\n  argmax LP             = {res['argmax_lp']:.3f}   (bord {res['theta_c']:.3f}, tol {res['tol']:.3f})")
    print(f"  argmax dwell curiosite= {res['argmax_dwell_lp']:.3f}")
    print(f"  argmax dwell erreur   = {res['argmax_dwell_error']:.3f}   (attendu cote chaos)")
    if res["argmax_lp_surrogate"] is not None:
        print(f"  argmax LP surrogate   = {res['argmax_lp_surrogate']:.3f}   (doit s'eloigner du bord)")
    print(f"  Spearman(LP, crit)    = {res['spearman']:+.3f}")
    print(f"  H1 LP@bord:{res['H1']}  H2 curiosite@bord:{res['H2']}  "
          f"H3 erreur->chaos:{res['H3']}  H4 random plat:{res['H4']}  H5 corr>=.5:{res['H5']}")
    print(f"  >>> H6 (H1 & H3) : {'PASS' if res['H6_pass'] else 'FAIL'}"
          + (f"   surrogate propre:{res['surrogate_clean']}" if res['surrogate_clean'] is not None else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    if args.smoke:
        subs = [LogisticMap(n=12)]
        seeds = [0]
        lp_kw = dict(budget=400, chunk=100, H=10, n_init=5)
        agent_kw = dict(pulls=120, chunk=15, H=10, n_init=4)
        do_sur = False
    else:
        subs = [LogisticMap(n=30), CoupledMapLattice(n=20), RandomBooleanNetwork()]
        seeds = list(range(5))
        lp_kw = dict(budget=1500, chunk=150, H=12, n_init=8)
        agent_kw = dict(pulls=None, chunk=20, H=12, n_init=6)
        do_sur = True

    t0 = time.time()
    out = {"meta": {"seeds": seeds, "smoke": args.smoke}, "substrates": []}
    passes = 0
    for sub in subs:
        print(f"\n[{time.time()-t0:6.1f}s] -> {sub.name} ...", flush=True)
        res = run_substrate(sub, seeds, lp_kw, agent_kw, do_surrogate=do_sur)
        print_report(res)
        out["substrates"].append(res)
        passes += int(res["H6_pass"])

    n = len(subs)
    print("\n" + "#" * 70)
    verdict = ("CONFIRMATION FORTE" if passes == n == 3 else
               "CONFIRMATION PARTIELLE" if passes >= 2 else
               "REFUTATION / NON DEMONTRE")
    print(f"VERDICT GLOBAL : {passes}/{n} substrats passent H6  ->  {verdict}")
    print("#" * 70)
    out["meta"]["passes"] = passes
    out["meta"]["n"] = n
    out["meta"]["verdict"] = verdict
    out["meta"]["seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresultats -> {args.out}   ({out['meta']['seconds']}s)")


if __name__ == "__main__":
    main()
