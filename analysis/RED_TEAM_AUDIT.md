# RED_TEAM_AUDIT.md — adversarial audit of the GBS / gap-law chain (2026-09-06)

Audited claims: (1) GBS = 2 at the coherent edge, all five optimizers; (2) κ_B = min(coherent edge,
κ_full + 2(1−β)); (3) formula-free gap×G2 = 2; (4) Adam/Muon = structured non-LTI boundary.
All numbers below computed fresh from `results/*` npz. Ranked most-damaging first.

## 1. SERIOUS — the "constant 2" is lr-dependent; "2" is a hottest-live-lr ceiling, not a plateau invariant
gapC = (κ_B − κ_full)/(1−β) within stationary same-(β,B) families across lr:
- HB β0.6 b8: 1.81 → 1.98 → 2.24 (lr 0.0025→0.0065), log-log slope **+0.22**
- HB β0.9 b32: 1.87 → 2.18 → 2.56 (lr 0.0015→0.005), slope **+0.26**
- HB β0.9 b128: **0.71 → 1.23 → 1.73 → 1.93** (lr 0.0015→0.011), slope **+0.51**
The registered ±30% band [1.4, 2.6] is roughly the spread the preflight's hottest-live-lr rule
produces. Honest restatement: **gap ≤ ~2(1−β), equality approached at the maximal live lr**; at b128
the "law" is a factor-2.7 lr effect. Also: the headline "C(128) = 21, CV 6%" was produced by same-day
per-cell drift/edge filtering — the first unfiltered scan printed **CV 0.36** for the same b128 pool
(β0.6-mem family at C≈6.4 and β0.99 lr-spread included). Filter choices were not pre-registered.
CB/CB2 confirmations remain genuine out-of-sample hits, but they are *conditioned on the same lr rule*.

## 2. SERIOUS — "formula-free G2" is near-tautological for the cells it passes on
For any LTI optimizer the step is by construction a fixed linear filter of past gradients, so the
measured gu0→su0 transfer **is** the known filter; G2 ≈ 1/(1−β) is an estimator identity, not
physics. Claim (3) therefore reduces to claim (2) plus instrument validation; CB2 is replication of
the (1−β) scaling, not a new universality. The only cells where G2 is genuinely formula-free — Adam
(0.8) and Muon (band-sensitive 4–7) — are exactly the cells where the law FAILS. "Universal
formula-free law" is overclaimed; correct statement: the (1−β) scaling holds for the LTI family.

## 3. SERIOUS (reframing risk) — "= 2" may be a soft O(1) fluctuation balance, not a sharp stability edge
Everything supporting the gap law is *plateau position*. No divergence bracket (ground-truth wall)
was ever run at ANY small-batch memory-ceiling cell (the wall/margin dataset covers coherent cells;
b8 rows were censored under 10.1(ii)). The lr-drift (finding 1), the Nesterov elevation (2.5–2.8),
and the β-trend (1.95 at β0.5 → 2.6 at β≥0.9) all fit "O(1) balance" as well as "edge constant 2".
**Decisive test (proposed, ~1 h): replay-brackets at c×lr around CB_hb09_b16** — a sharp onset just
above the plateau ⇒ real edge; smooth degradation ⇒ soft balance. This single experiment most
changes the interpretation.

## 4. SERIOUS caveat — Adam's "batch-independent 4.6" rests on a drifting estimator
On CB_adam_b16: `lam_full_w` late-window CV **0.35** with a **+24%/window trend** (10 warm power
iters may be under-converged; inc-r1 −0.50). The Adam gap carries ≥ ±15% systematic; "batch-
independent ≈ 4.6" is loose. (Raw-frame `lam_full` for Adam: CV 0.41, −86%/window — junk, as already
assumed; earlier raw-frame Adam gap numbers were correctly never used.)

## 5. MINOR / DEFUSED — motion-induced gap (reverse causality)
If parameter fluctuations inflated λ_B − λ_full, λ_B should track recent motion:
corr(λ_B, rolling RMS of s·u) = **+0.08 to +0.12** on HB β0.9/β0.3/β0.95 b8 and SGD b8 alike
(corr with loss +0.23–0.36 = ordinary curvature-loss coupling). No support for reverse causality.
Frozen-θ many-batch λ_B test still recommended (slow_sweep saves no checkpoints; needs a ~40-min
rerun with snapshots).

## 6. MINOR (reinterpretation) — GBS = 2 at the coherent edge is energy-balance bookkeeping
Exact stationarity in a locally quadratic basin forces E[gᵀs] + ½E[sᵀHs] = 0, i.e. ratio-of-means
= 2 for ANY optimizer; the per-step median ≈ 2 additionally needs single-mode dominance. So the
five-optimizer "GBS = 2" table certifies *stationary mode-dominated oscillation*, not a mysterious
transcendent constant. Consistent: SGD b8 (bulk-dominated, still descending) reads 1.85; Adam 2.05.
The nontrivial physics is that an edge/attractor exists — the value 2 is bookkeeping.

## 7. MINOR — cherry-picks to retract in prose (not in data)
- "Even SGD fits the gap law (1.87 at b8)": SGD's gap is 0.23 at b128 and 0.05 at b512 — under the
  min-law SGD's gap is unconstrained; the b8 agreement is a coincidence and was quoted as support.
- Preflight DC floor 0.4·2(1−β) shares the law's (1−β): it cannot manufacture a value sitting
  4–26× above the floor, but it does censor non-momentum optimizers (muon05_b8 nearly censored) —
  a domain-shaping selection, now documented.

## 8. MINOR — external validity is thin
One dataset subset (CIFAR-2048), MSE, two MLPs, mostly 1 seed/cell. Cross-arch support for the gap
law: mlp_l b32 ✓ (2.01), mlp_l b8 unresolved (≈1.2–1.4, nonstationary cells). ~20 candidate
laws/coordinates were tried and killed across this arc before the survivor; the survivor's strongest
evidence is the CB point predictions (6 gaps spanning 25×, median error ~15%) — that part is not
explainable by band-width + multiple comparisons; the "exactly 2" part is.

## Overall judgment
The program is finding structure, not fitting noise — but less than the current prose claims. Solid:
the ceiling exists, is stationary, scales as (1−β) at fixed (arch, batch, lr-rule) with point
predictions confirmed over a 25× range, and dissolves into the coherent edge as the curvature gap
closes. Overclaimed: "universal constant 2" (it is lr-conditioned, 1.8–2.6, and possibly a soft
balance); "formula-free" (tautological where it passes); "all five optimizers" for anything beyond
stationarity-bookkeeping GBS. The single most valuable experiment: **divergence brackets around one
small-batch memory ceiling (CB_hb09_b16)** — it decides edge vs balance, which is the difference
between a stability law and a fluctuation phenomenology.
