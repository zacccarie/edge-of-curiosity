"""Test decisif ρ(X,M) sur le CML : meme substrat, meme predicteur, OBSERVABLE different.
Si la cible de la curiosite (argmax LP) change avec la lentille, alors la frontiere
d'apprenabilite est une propriete du couple (systeme x observable), pas du substrat.

Prediction pre-enregistree (PREREG_LEARNABILITY, controle secondaire) :
  - lentille GROSSIERE (champ moyen, mode collectif lent) : chaos apprenable -> pic CHAOS
  - lentille FINE (deviation au champ moyen, turbulence)   : chaos non-apprenable -> pic BORD
"""
import numpy as np, time, json
import predictor
from experiment import spearman
from substrates import CoupledMapLattice

SEEDS = [0, 1, 2]
H = 16

sub = CoupledMapLattice(n=26)
thetas = sub.thetas
crit = np.mean([[sub.criticality_raw(t, s) for t in thetas] for s in SEEDS], axis=0)
r_c = sub.locate_edge(thetas, crit); peaked = sub.peaked_from_raw(crit)
binw = float(np.mean(np.diff(thetas)))

t0 = time.time()
print(f"CML  bord r_c={r_c:.3f}  binw={binw:.3f}  H={H}  (chaos = r eleve)")
print(f"{'lentille':>10} {'argmaxLP':>9} {'d=amax-rc':>10} {'spearman':>9} {'cible':>8}")
print("-" * 52)
out = {"r_c": r_c, "binw": binw, "H": H, "lenses": {}}
for lens in ["coarse", "full", "fine"]:
    lp = np.mean([[predictor.intrinsic_lp(sub, t, s, hid=16, budget=1200, chunk=150,
                                          H=H, n_init=8, lens=lens)[0]
                   for t in thetas] for s in SEEDS], axis=0)
    amax = float(thetas[int(np.argmax(lp))])
    d = amax - r_c
    sp = spearman(lp, peaked)
    cible = "BORD" if abs(d) <= 2 * binw else ("CHAOS" if d > 0 else "ORDRE")
    print(f"{lens:>10} {amax:9.3f} {d:+10.3f} {sp:+9.3f} {cible:>8}")
    out["lenses"][lens] = dict(argmax=amax, d=d, spearman=sp, cible=cible, lp=lp.tolist())

co, fi = out["lenses"]["coarse"], out["lenses"]["fine"]
shift = co["cible"] == "CHAOS" and fi["cible"] == "BORD"
print(f"\nDECISIF : grossiere->{co['cible']} , fine->{fi['cible']}  =>  "
      f"{'ρ(X,M) CONFIRME (la cible depend de la lentille)' if shift else 'pas de bascule nette'}")
out["rho_confirmed"] = bool(shift)
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open("results_lens_cml.json", "w"), indent=2)
print(f"-> results_lens_cml.json   ({out['seconds']}s)")
