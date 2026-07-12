"""
kappa_spec ESTIMATOR -- grep-certified FORMULA-FREE (KSPEC_DESIGN.md "OFFLINE ANALYSIS").
This file must contain NO optimizer transfer formula and must never use the momentum parameter:
the gain must FALL OUT of the measured gradient->step transfer, never be typed. Certify with:
  grep -nE '\(1[+-]beta\)|\(1\+2\*beta\)|1 *[+-] *beta|beta *\* *e|exp\(' experiments/kspec_estimator.py

Per cell (dense.npz from the patched slow_sweep runner, stride 1):
  - plateau window: measured rows where gu0 is finite (the runner freezes the fixed frame u0 at
    plateau start, so "first finite gu0" self-describes the window start); NaN rows dropped.
  - T_hat(w) = S_{gu,su}(w) / P_gu(w)   (Welch cross-spectrum / PSD -- LTI transfer gradient->step
    projected on the sign-aligned per-step top eigvec u_B; exact for SGD-family, no obs noise on gu).
  - spectral weight mu(w) = P_gu(w) (primary; P_dxu robustness) -- the measured operating-frequency
    distribution of the mode. NOT a point eval at arccos(r1) (edge-sensitive, LESSONS.md).
  - kappa_spec = median(lam_batch) * integral |T_hat(w)| dmu(w)
  - omega* centroid (reporting only), r1 = lag-1 autocorr of dxu (gate input, computed here but the
    GATE LABELS are assigned in kspec_gate.py before kappa_spec is looked at).
  - fixed-frame duplicate from (gu0, su0): must agree with in-frame at large batch.
  - split-half (first/second half of window) kappa_spec for stability.
"""
import os, json
import numpy as np
from scipy import signal


def _welch_transfer(x, y, nperseg):
    """|T_hat(w)| = |S_xy/P_xx| with Welch averaging; returns (freqs_rad, |T|, weight P_xx, coh)."""
    f, Pxx = signal.welch(x, detrend="constant", nperseg=nperseg)
    _, Pyy = signal.welch(y, detrend="constant", nperseg=nperseg)
    _, Sxy = signal.csd(x, y, detrend="constant", nperseg=nperseg)
    w = 2 * np.pi * f                       # rad/step, [0, pi]
    T = np.abs(Sxy) / np.maximum(Pxx, 1e-300)
    coh = np.abs(Sxy) ** 2 / np.maximum(Pxx * Pyy, 1e-300)
    return w, T, Pxx, coh


def _integral(T, weight):
    return float(np.sum(T * weight) / np.maximum(np.sum(weight), 1e-300))


def _r1(x):
    x = x - np.mean(x)
    d = float(np.dot(x, x))
    return float(np.dot(x[1:], x[:-1]) / d) if d > 0 else float("nan")


def analyze_cell(cell_dir, nperseg_cap=2048):
    z = np.load(os.path.join(cell_dir, "dense.npz"))
    meta = json.load(open(os.path.join(cell_dir, "meta.json")))
    gu, su, dxu = z["gu"], z["su"], z["dxu"] if "dxu" in z else None
    gu0, su0, lam = z["gu0"], z["su0"], z["lam_batch"]
    # plateau window = fixed-frame live (u0 frozen at plateau start by the runner)
    win = np.isfinite(gu0) & np.isfinite(gu) & np.isfinite(su) & np.isfinite(lam)
    if win.sum() < 512:
        return dict(cell=os.path.basename(cell_dir), ok=False, why=f"window too short ({win.sum()})")
    gu, su, gu0, su0, lam = gu[win], su[win], gu0[win], su0[win], lam[win]
    dxu = dxu[win] if dxu is not None else None
    n = len(gu)
    nperseg = int(min(nperseg_cap, 2 ** np.floor(np.log2(max(n // 6, 256)))))
    lam_med = float(np.median(lam))
    lr = meta["lr"]

    w, T, Pgg, coh = _welch_transfer(gu, su, nperseg)
    kspec = lam_med * _integral(T, Pgg)
    omega_star = _integral(w, Pgg)                      # PSD-weighted centroid, reporting only
    gain = _integral(T, Pgg) / lr                       # |T|/lr, dimensionless measured gain

    # robustness variants
    kspec_coh = lam_med * _integral(T, Pgg * (coh > 0.5))          # coherence-masked
    if dxu is not None and np.all(np.isfinite(dxu)):
        _, Pdx = signal.welch(dxu, detrend="constant", nperseg=nperseg)
        kspec_dxw = lam_med * _integral(T, Pdx)                    # increment-weighted
    else:
        kspec_dxw = float("nan")
    # fixed-frame duplicate
    wf, Tf, Pgg0, _ = _welch_transfer(gu0, su0, nperseg)
    kspec_fixed = lam_med * _integral(Tf, Pgg0)
    # split-half stability
    h = n // 2
    _, T1, P1, _ = _welch_transfer(gu[:h], su[:h], min(nperseg, 2 ** int(np.floor(np.log2(max(h // 4, 128))))))
    _, T2, P2, _ = _welch_transfer(gu[h:], su[h:], min(nperseg, 2 ** int(np.floor(np.log2(max(h // 4, 128))))))
    ks1, ks2 = lam_med * _integral(T1, P1), lam_med * _integral(T2, P2)

    return dict(cell=os.path.basename(cell_dir), ok=True,
                optn=meta["optn"], beta=meta["beta"], batch=meta["batch"], lr=lr,
                seed=meta.get("seed"), status=meta["status"], diverged=meta.get("diverged"),
                n_window=n, nperseg=nperseg,
                lam_med=lam_med, kappa_raw=lr * lam_med,
                r1_dxu=_r1(dxu) if dxu is not None else _r1(np.diff(np.cumsum(gu))),
                omega_star=omega_star, omega_star_over_pi=omega_star / np.pi,
                gain=gain,
                kappa_spec=kspec, kappa_spec_coh=kspec_coh, kappa_spec_dxw=kspec_dxw,
                kappa_spec_fixed=kspec_fixed, kappa_spec_h1=ks1, kappa_spec_h2=ks2)


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:]:
        print(json.dumps(analyze_cell(d), indent=1))
