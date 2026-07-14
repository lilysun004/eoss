"""
Tier-2 dataset assembler (ADDENDUM 8). DATA ONLY -- no fits here (tier2_fit.py, run in the
analysis session, does that). Builds one row per (cell, seed) joining:

  Y-side (intervention instrument, loss/lr observables only):
    c_onset_lo/hi   -- bracket: highest quiet c / lowest excited-or-dead c (excitation rule
                       ADDENDUM 5: max_loss > 3x plateau max, or died)
    c_death         -- lowest died c (or > max probed)
    margin          -- geometric midpoint of onset bracket, minus 1
    budget          -- c_death / c_onset_mid (lower bound if no death observed)

  X-side (passive instrument, undisturbed plateau):
    r1              -- lag-1 autocorr of dxu (from the cell's own kspec json)
    decoh           -- 1 - |r1|
    offpi           -- 1 - (PSD_gu fraction at omega > 3pi/4) over the plateau window
    cv2h            -- pool CV(h)^2 (fixed-u curvature noise), where a pool exists
    plus kappa_raw, kappa_spec, gain, optimizer, beta, batch for overlays/labels.

Writes kspec_results/tier2_dataset.json (+ .csv). Rerunnable as brackets land.
"""
import os, sys, json, glob
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
OUT = os.path.join(_REPO, "results", "kspec"); MS = os.path.join(OUT, "ms")
RES = os.path.join(_REPO, "kspec_results")


def plateau_lossmax(tag):
    z = np.load(os.path.join(OUT, tag, "dense.npz"))
    win = np.isfinite(z["gu0"])
    lo = z["loss"][win]
    n = len(lo)
    return float(np.max(lo[: max(2048, n // 3)]))     # early-window max (pre ring-down), robust


def offpi_weight(tag):
    from scipy import signal
    z = np.load(os.path.join(OUT, tag, "dense.npz"))
    win = np.isfinite(z["gu0"]) & np.isfinite(z["gu"])
    gu = z["gu"][win]
    if len(gu) < 512:
        return float("nan")
    nper = int(min(2048, 2 ** np.floor(np.log2(max(len(gu) // 6, 256)))))
    f, P = signal.welch(gu, detrend="constant", nperseg=nper)
    w = 2 * np.pi * f
    return float(1 - P[w > 3 * np.pi / 4].sum() / max(P.sum(), 1e-300))


def main():
    br = json.load(open(os.path.join(MS, "bracket.json"))) if os.path.exists(os.path.join(MS, "bracket.json")) else []
    bytag = {}
    for r in br:
        bytag.setdefault(r["tag"], []).append(r)
    rows = []
    for tag, runs in sorted(bytag.items()):
        try:
            base = plateau_lossmax(tag)
        except Exception:
            continue
        quiet, excited, died = [], [], []
        for r in runs:
            if r["died_at"] is not None:
                died.append(r["c"])
            elif r["max_loss"] > 3 * base:
                excited.append(r["c"])
            else:
                quiet.append(r["c"])
        if not (excited or died):
            onset_lo, onset_hi = (max(quiet) if quiet else float("nan")), float("nan")
        else:
            onset_hi = min(excited + died)
            onset_lo = max([q for q in quiet if q < onset_hi], default=1.0)
        onset_mid = float(np.sqrt(onset_lo * onset_hi)) if np.isfinite(onset_hi) else float("nan")
        c_death = min(died) if died else float("nan")
        cmax = max(r["c"] for r in runs)
        kj = os.path.join(MS, f"{tag}_kspec.json")
        k = json.load(open(kj)) if os.path.exists(kj) else {}
        cv2h = float("nan")
        pp = os.path.join(MS, f"{tag}_pool.npz")
        if os.path.exists(pp):
            h = np.load(pp)["pool"][:, 0, 0]
            if len(h) > 1:
                cv2h = float(h.var() / h.mean() ** 2)
        rows.append(dict(
            tag=tag, optn=k.get("optn"), beta=k.get("beta"), batch=k.get("batch"),
            c_onset_lo=onset_lo, c_onset_hi=float(onset_hi), c_onset_mid=onset_mid,
            margin=onset_mid - 1 if np.isfinite(onset_mid) else float("nan"),
            c_death=float(c_death), death_lower_bound=bool(not died), c_max_probed=cmax,
            budget=(c_death / onset_mid) if (died and np.isfinite(onset_mid)) else float("nan"),
            r1=k.get("r1_dxu"), decoh=1 - abs(k["r1_dxu"]) if "r1_dxu" in k else float("nan"),
            offpi=offpi_weight(tag), cv2h=cv2h,
            kappa_raw=k.get("kappa_raw"), kappa_spec=k.get("kappa_spec"), gain=k.get("gain")))
    os.makedirs(RES, exist_ok=True)
    json.dump(rows, open(os.path.join(RES, "tier2_dataset.json"), "w"), indent=1)
    cols = ["tag", "optn", "batch", "beta", "margin", "budget", "c_onset_lo", "c_onset_hi",
            "c_death", "death_lower_bound", "r1", "decoh", "offpi", "cv2h",
            "kappa_raw", "kappa_spec", "gain"]
    with open(os.path.join(RES, "tier2_dataset.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[tier2] {len(rows)} rows -> kspec_results/tier2_dataset.{{json,csv}}")
    for r in rows:
        print(f"  {r['tag']:28s} onset=({r['c_onset_lo']},{r['c_onset_hi']}) margin={r['margin'] if r['margin']==r['margin'] else float('nan'):+.3f} "
              f"budget={r['budget'] if r['budget']==r['budget'] else float('nan'):.2f}{'(lb)' if r['death_lower_bound'] else ''} "
              f"r1={r['r1'] if r['r1'] is not None else float('nan'):+.2f} offpi={r['offpi']:.2f} cv2h={r['cv2h']:.3f}")


if __name__ == "__main__":
    main()
