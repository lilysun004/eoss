# κ_spec pre-registered per-cell annotations — committed BEFORE any ladder cell is analyzed

Written while the ladder is running (2026-07-12), REVISED once (same session, still before any
ladder dense.npz was opened) after review caught the first draft mis-registering the mixed-cell
prediction. Only old-sweep data and pre-flight probes have been touched.

**Disclosures:** (1) the estimator plumbing smoke ran on pre-flight probe
`PRE_b8_beta0.9_lr0.002` (3000 steps, transient-heavy, not a ladder cell): κ_spec≈0.28,
gain≈0.92, r₁≈+0.25. The gate criterion predates it and uses no κ_spec. (2) The stationarity
table below was computed on OLD sweep cells only.

## The three-way prediction map (keyed on PLATEAU-NESS, not coherence class)

The first draft registered "mixed ⇒ predicted κ_spec < 2", which inverts the theory: the
spectral-integral hypothesis says a *plateaued* cell operating at mixed ω is marginal with
respect to its own broadband spectrum — the interpolation-band money case where only κ_spec can
read 2. Corrected registration:

- **(i) Plateaued cells, any ω (incl. mixed): predict κ_spec ≈ 2.** Strongest form of the
  hypothesis; mixed-ω plateaued cells (b128 headline) are its sharpest test.
- **(ii) Genuinely sub-plateau cells (κ non-stationary / still climbing): predict κ_spec < 2.**
- **(iii) Endpoint cells (coherent b2048 / DC β0.99-b8 if the blind gate labels it so):
  κ_spec ≈ 2 AND the measured gain must land at the known endpoint values as a cross-check**
  (gain ≈ 1/(1+β) at ω=π, ≈ 1/(1−β) at ω=0 — checked on the gate/anchor side; the estimator
  never contains these).

**Stationarity criterion (fixed now, computed from raw κ only, blind to κ_spec):** split the
plateau window into thirds; plateaued iff |median κ(third 3)/median κ(third 1) − 1| < 0.10.

**Stationarity measured on OLD equivalents (2026-07-12):** b8 −4.8/−4.0%, b32 −0.9/−0.3%,
b128 +2.7/+1.5%, b512 +0.7%, b2048 +0.0% → ALL stationary. So the entire β0.9 ladder is
expected class (i): **the strong form is on the line at every batch**, including b8.

**The b8 anomaly, registered rather than classified away:** b8 β0.9 plateaus at κ≈0.34, 70%
ABOVE its DC-law value 2(1−β)=0.20, with mixed r₁≈+0.25. Readings: (a) it is not in the pure DC
regime — its marginality is at intermediate ω (the κ_spec story, prediction ≈2); (b) the probe
preview (gain≈0.92 where ≈5.9 would be needed) hints the strong form may FAIL here — if the full
cell confirms gain≈1 with a stationary κ, then b8 is a plateaued cell that is NOT spectrally
marginal, and class (i) is falsified at small batch → the honest headline becomes "κ_spec = 2 on
the coherent-to-mixed branch only" or a min()-structure survival. Either outcome is informative
and will be reported as measured. (c) A harness-vs-paper-law disagreement would also hit the
Nesterov anchor — check the anchor before interpreting trio κ_spec (ruling 3 below).

## Secondary registered test — the sub-2 axis made quantitative (non-tautological form)

"κ_spec shortfall = gain-corrected raw shortfall" is circular (κ_spec/2 ≡ λ/λ_marg by
definition). The quantitative version registered instead: **κ_spec must agree cell-by-cell with
median plateau GBS** — two independent marginality instruments (HVP quadratic form s'Hs/(−g's)
vs gradient→step transfer spectroscopy), both = 2 at any binding edge, and predicted to fall
short by the SAME factor on any sub-marginal cell. Report: per-cell (GBS_med, κ_spec) with
correlation and slope across the ladder; prediction slope ≈ 1 through the origin-anchored fit.
Caveat registered: old stride-2 large-batch data shows GBS_med 0.61 at the b2048 at-edge cell —
suspected stride-2 phase-locking of a period-2-alternating GBS (same aliasing family as the ω=π
signal); the stride-1 ladder is the clean read, and if stride-1 GBS_med at b2048 still ≠ 2 that
is itself a finding about GBS phase-sensitivity at coherent cells, to be reported not patched.

## Interpretation rulings (fixed before data)

1. The gate's "OR independent attractor/at-edge evidence" clause will NOT be invoked; park-test
   attractor evidence is regulation evidence, not marginality evidence.
2. The COMMITTED instrument layer is unchanged: gates.json labels per kspec_gate.py, and the
   KSPEC_DESIGN PASS criteria (median κ_spec ∈ [1.8,2.2] AND CV(κ_spec) < 0.5·CV(κ_raw) on
   gate=at-edge cells) are evaluated exactly as committed. The three-way map above is the
   theory-faithful prediction layer evaluated ALONGSIDE it. Both verdicts get reported;
   neither is dropped post-hoc.
3. Nesterov trio anchor (STOP gate, kspec_ladder.py --anchor): Nesterov b2048 raw plateau κ
   within 15% of 2(1+β)/(1+2β) (β=0.9 → 1.357), evaluated BEFORE any trio κ_spec. Small-batch
   position reported informationally only. Anchor fail ⇒ stop and debug, no trio κ_spec claims.

## Per-cell table (predictions; blind gate still assigns the committed labels on NEW data)

β0.9: coherent law 3.80, DC law 0.20. β0.99: coherent 3.98, DC 0.02.

| cell | lr | old κ | stationary | expected class | prediction |
|---|---|---|---|---|---|
| L_b8_beta0.9    | 0.0020 | 0.34 | yes | (i) plateaued-mixed | κ_spec ≈ 2 (strong form; probe hints fail → informative) |
| L_b32_beta0.9   | 0.0050 | 0.63 | yes | (i) plateaued-mixed | κ_spec ≈ 2 |
| L_b128_beta0.9  | 0.0060 | 2.10 | yes | (i) plateaued-mixed | κ_spec ≈ 2 — HEADLINE interpolation test |
| L_b512_beta0.9  | 0.0080 | 3.66 | yes | (i)+(iii) coherent | κ_spec ≈ 2, gain → 1/(1+β) |
| L_b2048_beta0.9 | 0.0065 | 3.81 | yes | (i)+(iii) coherent endpoint | κ_spec ≈ 2, gain → 1/(1+β), fixed-frame ≡ in-frame |
| L_b8_beta0.99   | bisect TBD | TBD | TBD | (iii) DC endpoint if gated so | κ_spec ≈ 2 from raw κ ≈ 0.02–0.03; ω-resolution flag applies |

If L_b8_beta0.99 does not gate at-edge AND is non-stationary, the DC endpoint goes untested
(untested ≠ failed; the report must say which).
