"""Test relationnel : l'horizon de rollout H (bouton de l'OBSERVATEUR seul)
fait-il glisser argmax(LP) le long de l'axe ordre->chaos ?

Prediction miroir pre-enregistree (PREREG_LEARNABILITY.md) :
  - CML : H croit -> pic chaos -> bord  (d(H) decroit)
  - Logistique : H croit -> pic bord ; H decroit -> pic chaos  (d(H) croit avec H... non :
    d signe = argmax - r_c, chaos = r eleve. Logistique : a H court le chaos est apprenable
    -> d grand ; a H long -> d ~ 0. Donc d decroit aussi avec H, mais part de plus haut.)
Le point cle : pour les DEUX, d(H) doit DECROITRE avec H (le pic se rapproche du bord
quand l'horizon allonge), et la logistique l'atteint plus tot que le CML.
"""
import numpy as np, time, json
import predictor
from experiment import spearman
from substrates import LogisticMap, CoupledMapLattice

SEEDS = [0, 1, 2]
HS = [1, 2, 4, 8, 16, 32]
BUDGET, CHUNK, NINIT, HID = 1200, 150, 8, 16


def edge_and_peaked(sub):
    thetas = sub.thetas
    crit = np.mean([[sub.criticality_raw(t, s) for t in thetas] for s in SEEDS], axis=0)
    return thetas, sub.locate_edge(thetas, crit), sub.peaked_from_raw(crit)


def lp_profile(sub, H):
    thetas = sub.thetas
    return np.mean([[predictor.intrinsic_lp(sub, t, s, hid=HID, budget=BUDGET,
                                            chunk=CHUNK, H=H, n_init=NINIT)[0]
                    for t in thetas] for s in SEEDS], axis=0)


def run(sub):
    thetas, r_c, peaked = edge_and_peaked(sub)
    binw = float(np.mean(np.diff(thetas)))
    rows = []
    print(f"\n=== {sub.name.upper()}  bord r_c={r_c:.3f}  binw={binw:.3f}  (chaos = r {'eleve' if sub.chaos_side=='high' else 'faible'}) ===")
    print(f"{'H':>4} {'argmaxLP':>9} {'d=amax-rc':>10} {'spearman(LP,crit)':>18}")
    for H in HS:
        lp = lp_profile(sub, H)
        amax = float(thetas[int(np.argmax(lp))])
        d = amax - r_c
        sp = spearman(lp, peaked)
        rows.append(dict(H=H, argmax=amax, d=d, spearman=sp, lp=lp.tolist()))
        print(f"{H:>4} {amax:9.3f} {d:+10.3f} {sp:+18.3f}")
    # monotonie d(H) vs H
    dvals = [r["d"] for r in rows]
    sp_Hd = spearman(HS, dvals)
    sp_Hsp = spearman(HS, [r["spearman"] for r in rows])
    print(f"  Spearman(H, d) = {sp_Hd:+.3f}   (attendu < 0 : pic -> bord quand H croit)")
    print(f"  Spearman(H, corr) = {sp_Hsp:+.3f}   (attendu > 0 : LP recorrele au bord quand H croit)")
    return dict(name=sub.name, r_c=r_c, binw=binw, chaos_side=sub.chaos_side,
                thetas=[float(t) for t in thetas], rows=rows,
                spearman_H_d=sp_Hd, spearman_H_corr=sp_Hsp)


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    for sub in [LogisticMap(n=30), CoupledMapLattice(n=26)]:
        out[sub.name] = run(sub)
    # verdict pre-enregistre
    cml, log = out["cml"], out["logistic"]
    L1 = cml["spearman_H_d"] <= -0.6
    L3 = log["spearman_H_d"] <= -0.6      # les deux doivent decroitre ; logistique part plus haut
    L4 = (cml["spearman_H_corr"] >= 0.5) and (log["rows"][-1]["spearman"] > log["rows"][0]["spearman"])
    print("\n" + "#"*64)
    print(f"L1 CML pic->bord (Sp(H,d)<=-.6): {L1}  [{cml['spearman_H_d']:+.3f}]")
    print(f"L3 LOG pic->bord (Sp(H,d)<=-.6): {L3}  [{log['spearman_H_d']:+.3f}]")
    print(f"L4 bascule corr LP->bord avec H: {L4}")
    verdict = ("CONFIRMATION FORTE relationnel" if (L1 and L3 and L4)
               else "PARTIEL" if (L1 or L3) else "REFUTATION")
    print(f"VERDICT : {verdict}")
    print("#"*64)
    out["verdict"] = dict(L1=bool(L1), L3=bool(L3), L4=bool(L4), verdict=verdict,
                          seconds=round(time.time()-t0, 1))
    json.dump(out, open("results_learnability.json", "w"), indent=2)
    print(f"\n-> results_learnability.json   ({out['verdict']['seconds']}s)")
