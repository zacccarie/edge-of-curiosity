"""Verification adversariale du resultat positif logistique.
Le pic de LP au bord survit-il a la variation du plancher FLOOR, de la capacite
(hid) et du budget ? Si oui -> robuste. Si seulement a FLOOR=0.05 -> artefact."""
import numpy as np
import predictor
from substrates import LogisticMap
from experiment import spearman

sub = LogisticMap(n=30)
seeds = [0, 1, 2]
thetas = sub.thetas
crit = np.mean([[sub.criticality_raw(t, s) for t in thetas] for s in seeds], axis=0)
theta_c = sub.locate_edge(thetas, crit)
peaked = sub.peaked_from_raw(crit)
binw = float(np.mean(np.diff(thetas)))

print(f"bord theta_c = {theta_c:.3f}   (tol = 1.5*binw = {1.5*binw:.3f})")
print(f"{'FLOOR':>6} {'hid':>4} {'budget':>7} | {'argmaxLP':>9} {'|dist|':>7} {'<=tol':>6} {'spearman':>9}")
print("-" * 60)
for floor in [0.02, 0.05, 0.10, 0.20]:
    for hid in [12, 16, 24]:
        budget = 1500
        predictor.FLOOR = floor
        lp = np.mean([[predictor.intrinsic_lp(sub, t, s, hid=hid, budget=budget,
                                              chunk=150, H=12, n_init=8)[0]
                       for t in thetas] for s in seeds], axis=0)
        amax = float(thetas[int(np.argmax(lp))])
        dist = abs(amax - theta_c)
        ok = "yes" if dist <= 1.5 * binw else "no"
        print(f"{floor:6.2f} {hid:4d} {budget:7d} | {amax:9.3f} {dist:7.3f} {ok:>6} "
              f"{spearman(lp, peaked):+9.3f}")
