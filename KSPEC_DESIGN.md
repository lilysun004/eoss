# κ_spec — the universal marginality scalar: pre-registration + runner patch spec

**The hypothesis (paper-closing).** The two known momentum plateaus are one closed-loop marginality
condition λ·|T(ω\*)| = 2 evaluated at two operating frequencies of the unstable mode:
- ω\* = π (coherent period-2) → |T| = η/(1+β) → λ = 2(1+β)/η  (large-batch plateau)
- ω\* = 0 (DC / decoherent) → |T| = η/(1−β) → λ = 2(1−β)/η  (noise-dominated plateau)

So the **path-computable universal scalar, no optimizer math**:
`κ_spec = λ_B · |T̂(ω*)|  =  2  at all batches/β`, where T̂ = empirically-identified gradient→step
transfer function and ω\* = realized operating frequency of the mode. R is then just the **decoherence
parameter that selects ω\*** — which explains R's half-collapse (ordinal proxy for ω\*, not the gain).

## BANKED (confirmed free, no rerun needed)
The ω\* migration is REAL and measured (differenced x_t = u·(θ−EMA), increment lag-1 autocorr r₁):
**r₁ goes +0.95 (small batch, DC) → −0.99 (b2048 β0.9, clean period-2)**; ω\*/π ≈ 0.1 → 0.95 up the
ladder. Endpoint cell b2048 β0.9: r₁=−0.99 → ω\*≈π → κ_spec = 3.70/(1+β) = **1.95 ≈ 2**. Retro-explains
the old AR-pole phases (SGD_b2048 phase π). **Status: mechanism confirmed; quantitative κ_spec=2 pending
this rerun.** The rerun's job is NOT "does ω\* migrate" (done) — it is **"does κ_spec = 2 hold
quantitatively at the cells where it should."**

## PRE-REGISTERED marginality gate + pass criteria (fixed BEFORE κ_spec is computed — no post-hoc)
Danger: "sub-edge cells shouldn't hit 2 anyway" is a post-hoc escape hatch. Close it with a BLIND gate:
- **Gate (independent instrument, applied before κ_spec):** a cell is "at its frequency-edge" iff its
  raw plateau κ is within 15% of the paper's two-regime plateau law for its (β, regime) — i.e. near
  2(1+β) if coherent (r₁<−0.3) or near 2(1−β) if DC (r₁>+0.3) — OR it has independent attractor/at-edge
  evidence. Cells failing the gate are pre-labeled "sub-edge" BEFORE seeing κ_spec.
- **Two-sided prediction (unescapable):** κ_spec ≈ 2 on GATED (at-edge) cells; κ_spec < 2 on sub-edge
  cells. If the paper's claim (BS *plateaus* at 2(1±β)) is right, most plateaued cells pass the gate, so
  a κ_spec failure on a plateaued/gated cell is a **genuine refutation**, not excusable.
- **PASS:** on gated cells, median(κ_spec) ∈ **[1.8, 2.2]** AND CV(κ_spec) < **0.5·CV(raw κ)**. Both,
  not either.

## RUNNER PATCH (the ONLY missing bit is SIGN — unsigned cosines destroyed the phase, unrecoverably)
Patch `slow_sweep.py`'s measurement block. Add SIGNED per-step columns (the runner already computes the
per-step top eigvec u via warm power iteration — use it directly, NOT M.cosabs which takes |·|):
- **in-frame** (each step projected on ITS OWN u_B): `gu = dot(gd, u)`, `su = dot(s, u)`,
  `mu = dot(m, u)` — signed.
- **increment**: `dxu = dot(step_applied, u)` (= s·u for SGD-family; log independently as cross-check).
- **fixed-frame** (frozen u0 snapshot at plateau start): `gu0 = dot(gd, u0)`, `su0 = dot(s, u0)` — for
  the large-batch frame-validation (in-frame and fixed-frame MUST agree at b2048 where u is static;
  disagreement = estimator bug).
- keep λ_B, cos_uu as-is. Snapshot u0 once when the plateau/record phase begins.
Add these to the FIELDS list + dense.npz save. Everything else identical. **Compute nothing clever
online — log raw signed primitives; all spectra/T̂/κ_spec are offline** (re-runnable as the estimator
is refined). Apply the **liveness-bisect pre-flight** (standing rule — 4th time a designed grid would
lose cells to the live-lr window; bisect lr down from canonical until non-diverging AND non-crawling).

## OFFLINE ANALYSIS (no formula plugged in — grep-certify)
1. ω-distribution: Welch PSD of the in-frame mode signal (use g·u or the increment, NOT θ·u which is
   drift-dominated). |T̂(ω)| from the **cross-spectrum** of (g·u, s·u).
2. κ_spec = λ_B · ∫|T̂(ω)| dμ(ω) over the measured spectral density μ (spectral INTEGRAL, not a point
   eval at the fragile arccos(r₁) centroid — arccos is hypersensitive at the DC end where the (1−β)
   gain lives).
3. **Open-loop validation** (kills closed-loop-bias worry, free): feed the recorded g·u series through
   the momentum recursion offline (exact for SGDM) and check T̂ matches the closed-loop estimate.
4. **Grep-certify** the analysis code contains no `(1+beta)`, `(1-beta)`, `(1+2*beta)`, or any optimizer
   transfer formula. The (1±β)/(1+2β) must FALL OUT of measured T̂, never be typed.

## CELLS (~15-20, targeted — NOT a re-sweep; gate a wide (β,B) collapse figure on THIS result)
- **Interpolation ladder FIRST:** SGD-Momentum β0.9, b8 → b32 → b128 → b512 → b2048, 2 seeds, live-lr
  bisect. **Money row = b128–b512** (preview ω\*/π ≈ 0.5–0.76 = genuinely mixed spectrum — where the
  point-frequency formula and raw κ both fail but the spectral-integral κ_spec can read 2).
- **β0.99 small-batch cell (b8) at its validated live lr — REQUIRED, the sharpest single test:** its
  predicted gain η/(1−β) = 100η is a 100× correction; if in-frame κ_spec lands ≈2 from a raw κ ≈ 0.03,
  that one cell is worth the ladder rhetorically.
- **Nesterov trio (immediately after, same patched runner):** needs a Nesterov optimizer added to
  `utils/optimizer.py` (with `compute_step_direction`). (a) SANITY ANCHOR: replicate the paper's
  thresholds in-harness — large-batch 2(1+β)/(η(1+2β)), small-batch noise value; if our setup doesn't
  reproduce their thresholds, STOP. (b) THE TEST: κ_spec = 2 on the same cells with zero Nesterov math
  (a third, structurally-different threshold falling out of measured T̂ = the "no optimizer math" claim
  demonstrated). Cells: Nesterov β0.9 at b8, b2048, + one mid-batch, 2 seeds.

## IMPLEMENTATION STATUS (staged; launch at session reset)
- [ ] slow_sweep.py signed-projection patch (spec above) — localized to the measurement block + FIELDS.
- [ ] Nesterov optimizer in utils/optimizer.py.
- [ ] p1-style driver for the 15-20 cells with liveness-bisect pre-flight.
- [ ] offline kspec analysis (Welch T̂ + spectral-integral + open-loop validation + grep-certify).
SUMMARY untouched until the gated ladder result is in. Wide-grid collapse figure gated on ladder PASS.
