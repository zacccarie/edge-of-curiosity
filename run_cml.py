"""Run CML seul (r-sweep + Lyapunov), iteration rapide vers H6."""
import json, time
from substrates import CoupledMapLattice
from experiment import run_substrate, print_report

sub = CoupledMapLattice()
seeds = list(range(5))
lp_kw = dict(budget=1500, chunk=150, H=12, n_init=8)
agent_kw = dict(pulls=None, chunk=20, H=12, n_init=10)

t0 = time.time()
res = run_substrate(sub, seeds, lp_kw, agent_kw, do_surrogate=True)
print_report(res)
json.dump(res, open("results_cml_r.json", "w"), indent=2)
print(f"\n{time.time()-t0:.1f}s")
