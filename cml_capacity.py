"""Test mecanistique : l'echec CML vient-il de la capacite du predicteur ?
Hypothese : LP-pique-au-bord exige que la capacite suffise a maitriser le bord
mais PAS le chaos. Si vrai, augmenter hid deplace argmax(LP) du chaos vers le bord.
Dose-reponse sur hid (intrinseque seul, 3 seeds, pas d'agents)."""
import numpy as np, time
import predictor
from substrates import CoupledMapLattice
from experiment import spearman

sub = CoupledMapLattice()
thetas = sub.thetas; seeds = [0, 1, 2]
crit = np.mean([[sub.criticality_raw(t, s) for t in thetas] for s in seeds], axis=0)
theta_c = sub.locate_edge(thetas, crit); peaked = sub.peaked_from_raw(crit)
binw = float(np.mean(np.diff(thetas)))
t0 = time.time()
print(f"bord theta_c={theta_c:.3f}  tol={1.5*binw:.3f}  (cote chaos = r eleve)")
print(f"{'hid':>4} {'argmaxLP':>9} {'|dist|':>7} {'<=tol':>6} {'spearman':>9}")
print("-" * 48)
for hid in [16, 48, 96, 160]:
    lp = np.mean([[predictor.intrinsic_lp(sub, t, s, hid=hid, budget=1500,
                                          chunk=150, H=12, n_init=8)[0]
                   for t in thetas] for s in seeds], axis=0)
    amax = float(thetas[int(np.argmax(lp))])
    ok = "yes" if abs(amax - theta_c) <= 1.5 * binw else "no"
    print(f"{hid:4d} {amax:9.3f} {abs(amax-theta_c):7.3f} {ok:>6} {spearman(lp,peaked):+9.3f}")
print(f"\n{time.time()-t0:.1f}s")
