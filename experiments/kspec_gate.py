"""
BLIND marginality gate (pre-registered in KSPEC_DESIGN.md BEFORE kappa_spec was computed).
This file is the INDEPENDENT instrument side -- it MAY use the paper's two-regime plateau law
(that is its job); the kappa_spec estimator (kspec_estimator.py) is the formula-free side.

Gate, verbatim from KSPEC_DESIGN.md:
  a cell is "at its frequency-edge" iff its raw plateau kappa is within 15% of the two-regime
  plateau law for its (beta, regime): near 2(1+beta) if coherent (r1 < -0.3), near 2(1-beta) if
  DC (r1 > +0.3). Cells failing the gate are pre-labeled "sub-edge" BEFORE seeing kappa_spec.
  (Mixed cells, |r1| <= 0.3, have no plateau-law prediction -> pre-labeled sub-edge/mixed here;
  the two-sided prediction applies to the labels as assigned, no post-hoc reshuffling.)

Inputs: ONLY raw plateau kappa (lr * median lam_batch over the plateau window) and r1 (lag-1
autocorr of the applied-step projection dxu). No kappa_spec, no T_hat.
Writes results/kspec/gates.json; refuses to overwrite (the gate is assigned once, blind).
"""
import os, sys, json
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO, "results", "kspec")


def _plateau_stats(cell_dir):
    z = np.load(os.path.join(cell_dir, "dense.npz"))
    meta = json.load(open(os.path.join(cell_dir, "meta.json")))
    gu0, lam, dxu = z["gu0"], z["lam_batch"], z["dxu"]
    win = np.isfinite(gu0) & np.isfinite(lam) & np.isfinite(dxu)
    if win.sum() < 512:
        return None
    kappa = meta["lr"] * float(np.median(lam[win]))
    x = dxu[win] - np.mean(dxu[win])
    r1 = float(np.dot(x[1:], x[:-1]) / np.dot(x, x))
    return meta, kappa, r1


def gate_cell(cell_dir):
    st = _plateau_stats(cell_dir)
    if st is None:
        return dict(cell=os.path.basename(cell_dir), gate="unusable")
    meta, kappa, r1 = st
    beta = meta["beta"]
    if r1 < -0.3:
        regime, pred = "coherent", 2 * (1 + beta)
    elif r1 > 0.3:
        regime, pred = "DC", 2 * (1 - beta)
    else:
        regime, pred = "mixed", None
    if pred is not None and abs(kappa / pred - 1) <= 0.15:
        gate = "at-edge"
    else:
        gate = "sub-edge"
    return dict(cell=os.path.basename(cell_dir), beta=beta, batch=meta["batch"],
                kappa_raw=kappa, r1=r1, regime=regime, plateau_pred=pred, gate=gate)


def main(cell_dirs):
    path = os.path.join(OUT, "gates.json")
    if os.path.exists(path):
        print(f"[gate] {path} exists -- gate already assigned (blind, once). Not overwriting.")
        return
    gates = [gate_cell(d) for d in cell_dirs]
    json.dump(gates, open(path, "w"), indent=1)
    for g in gates:
        print(f"  {g['cell']:28s} kappa={g.get('kappa_raw', float('nan')):7.3f} "
              f"r1={g.get('r1', float('nan')):+.2f} {g.get('regime', '-'):9s} "
              f"pred={g.get('plateau_pred')} -> {g['gate']}")


if __name__ == "__main__":
    main(sys.argv[1:])
