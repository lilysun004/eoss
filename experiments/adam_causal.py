"""Direct causal test on Adam: perturb-and-relax (does a kick persist=marginal or decay=
metastable?) for Adam b8, with SGD b8 (marginal control) and SGDM b8 (metastable reference)
in the same run. Kick along both u_hess (raw top eigvec) and u_step (step-PCA = optimizer's
own geometry, parameter-free -- the honest direction for a preconditioned optimizer)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DATASETS', '/Users/xq/Desktop/moonshot/eoss/datasets')
os.environ.setdefault('EOSS_SKIP_CHECKSUM', '1')
import numpy as np, experiments.perturb_relax as PR
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'adam_causal')
os.makedirs(OUT, exist_ok=True)
cells = [
    ("SGD_b8",   "SGD",          {},                            0.01,  16000),   # marginal control
    ("Adam_b8",  "Adam",         {"beta1":0.9,"beta2":0.99},    0.001, 20000),   # the question
    ("SGDM_b8",  "SGD-Momentum", {"beta":0.9},                  0.002, 14000),   # metastable ref
]
res=[]
for tag, optn, params, lr, steps in cells:
    print(f"\n=== {tag}: {optn} lr={lr} ===", flush=True)
    st = PR.train_plateau(optn, params, 8, lr, steps)
    if st is None:
        print(f"  {tag}: diverged"); res.append({"tag":tag,"diverged":True}); continue
    for uname,u in [("u_hess",st['u_hess']),("u_step",st['u_step'])]:
        r = PR.analyze(st, u, tag)
        lin = next((a for a in r['amps'] if a['amp_over_scale']>0.5), r['amps'][-1])
        esc = next((a['amp_over_scale'] for a in r['amps'] if a['escaped']), None)
        rec = {"tag":tag,"optimizer":optn,"kick_dir":uname,"gamma_relax":lin['gamma'],
               "escape_x_natural":esc,"scale":r['scale']}
        res.append(rec)
        print(f"  [{uname}] gamma_relax={lin['gamma']:+.4f}/step  escape={esc if esc else '>16'}x", flush=True)
        json.dump(res, open(os.path.join(OUT,'adam_causal.json'),'w'), indent=2)
print("\n===== VERDICT: gamma_relax ~0 => MARGINAL (persists); gamma<0 => METASTABLE (decays) =====")
for r in res:
    if r.get('diverged'): print(f"{r['tag']} diverged"); continue
    print(f"  {r['tag']:9s} [{r['kick_dir']:6s}] gamma_relax={r['gamma_relax']:+.4f}")
