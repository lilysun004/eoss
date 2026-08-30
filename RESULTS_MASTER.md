# RESULTS_MASTER.md — the one-stop synthesis (2026-08-02)

**What this is:** the single entry point for the κ_spec / κ_ms / Tier-2 program on branch `lily`.
Everything here is concrete and sourced; deeper detail lives in the docs listed in [§8](#8-document-index).
Status labels: **SETTLED** (registered gate passed/failed, written up), **DATA-ONLY** (numbers exist,
registered adjudication not yet run), **REFUTED** (registered kill).

Setup: CIFAR-10 subset (num_data=2048), MSE loss, CPU. Two MLPs: `mlp_s` (789k params, the main
instrument-development architecture) and `mlp_l` (512×4, 2.37M, the replication battery).
All estimators formula-free and grep-certified; all predictions registered before data
(`KSPEC_PREREG_ANNOTATIONS.md`, 10 addenda).

> **⚠️ 2026-08-30 UPDATE — GOLD sweep supersedes the per-cell numbers below.** One run per
> (optimizer × batch), κ_spec AND GBS from the same run/window, health-masked (float32-death fix):
> **`GOLD_RESULTS.md`** is now the table of record. Three changes of substance:
> **(1)** the heavy-ball large-batch "GBS ≠ 2" readings were a numerically-dead-run artifact
> (`analysis/HB_B2048_GBS_PROBE.md`) — on healthy windows **GBS = 2.00 at the coherent edge for all
> five optimizers, including Muon** (κ_spec-invalid there), raw thresholds spanning 280×;
> **(2)** the small-batch ceiling is now explained one level deeper: **κ_B = min(coherent edge,
> C(B,arch)·(1−β))** — memory enters purely as effective lr; verified out-of-sample by the gold
> SGDM/SGD/Nesterov columns (`analysis/MEMORY_EDGE_LAW.md`); C(B) ≈ 3.3/6.2/12/21 at b8–b128 (mlp_s)
> is the remaining unexplained constant; **(3)** the registered one-coordinate collapse test FAILED
> for every candidate (batch, r₁, u-rotation, cos(s,g); `analysis/collapse_gbs_gold.png`) — there is
> no single-coordinate universal GBS curve; the min-law above is the standing two-coordinate structure.

---

## 1. The headline: what "universal stable quantity or law" survived

The original north star — *one scalar that equals 2 at every training plateau, for every optimizer
and batch size, at some moment order* — **does not exist. That is now a measured result, not a
suspicion** (first moment: strong form refuted at b8/b32/b128/β0.99; second moment: Reading A dead
everywhere except large batch; see §5 graveyard).

What survived is a **two-tier structure**, each half universal in a different sense:

**Tier 1 — spectral marginality (SETTLED on mlp_s).**
κ_spec ≡ λ_B·|T̂(ω*)| = **2 wherever the closed loop is phase-coherent (r₁ → −1)**, for every
optimizer tested, with the optimizer's transfer gain falling out of the measurement rather than a
formula. Four raw thresholds spanning a factor of 25 (κ_raw 1.36 → 34) all collapse to 2:

| optimizer (b2048, mlp_s) | seeds | κ_raw threshold | measured gain (theory) | κ_spec |
|---|---|---|---|---|
| SGDM β=0.9 | 5 | 3.77–3.82 | 0.526 (1/(1+β) = 0.526) | **1.988** [1.981, 2.007] |
| Nesterov β=0.9 | 5 | 1.357 (= 2(1+β)/(1+2β), ratio 1.000) | 1.474–1.476 ((1+2β)/(1+β) = 1.4737) | **2.002** [2.001, 2.003] |
| Adam β₁=0.9 | 5 | 34.1–34.2 | 0.0568 (no formula; ideal-EMA 0.0526, +8% systematic) | **1.940** [1.936, 1.942] |
| SGD | (mlp_l) | 1.97 | 1.0000 | **1.970** |

Blind pre-registered gate: **PASS** — median 2.007, CV 0.006 (criterion < 0.009); every
sub-edge-labeled cell < 2 (two-sided prediction held). Details: `KSPEC_RESULTS.md` Headline 1.

Where coherence is absent, κ_spec is *not* 2 and the deficit is graded by decoherence
(mlp_s SGDM: b8 0.33 → b32 0.54 → b64 1.04 → b128 1.52 → b512 2.00 → b2048 2.0; open-loop
reconstruction R² decays +1.00 → −43 over the same ladder). All of these cells are stationary
plateaus — so plateau ≠ marginal. `KSPEC_RESULTS.md` Headline 2.

**Tier 1b — the mean-square wall (SETTLED on mlp_s).**
Every plateau, including the deeply sub-marginal ones, sits **inside a finite, measurable
mean-square stability wall**, located by ground-truth intervention (onset brackets: rerun real
training from bitwise-replayed checkpoints at c×lr). The wall is close everywhere — c* ∈
[1.0, ~1.4] — even where κ_raw is 60× below the first-moment ceiling. At the coherent end the wall
coincides with κ_ms = 2 exactly (γ₂ frozen-cocycle: 1.99 at b2048, 1.97 at b512, 2.01 nest b2048).
`KSPEC_RESULTS.md` Headline 3.

**Tier 2 — the margin law (PROVISIONAL, refit required).**
How far below the wall a plateau parks is **optimizer-independent** — the registered kill-test
(optimizer one-hot dummies) came back p = 0.72, so margins do not know the optimizer's name.
The partial functional form found: **margin ≈ 0.54·√CV(h)²** (noise-amplitude scaling), R² = 0.59,
LOOCV RMSE 0.14, through the origin. ⚠️ **This fit predates the ADDENDUM 10.1(ii) censoring
guard, which subsequently censored its highest-leverage anchor row (`L_b8_beta0.9_s0`, cv2h 0.74)
plus two more b8 rows. The 0.54, the R², and the "budget ≈ 1.03 razor-thin" anomaly all need
re-derivation.** The kill-test conclusion (optimizer-independence) is expected to survive — it
never rested on one row — but the refit is registered as analysis-session work and has not been run.
`KSPEC_RESULTS.md` Tier-2 verdict; ADDENDUM 8; commit `9c6a709`.

One-paragraph version: **the edge of stochastic stability is a spectral-marginality law (κ_spec = 2)
that holds universally in its coherent domain, backed everywhere by a mean-square wall whose
distance from the operating point is set by curvature-noise amplitude, not by optimizer identity.
The "universal constant" is 2; the universal *law* is two-tier; the coefficient 0.54 is the part
still in flux.**

---

## 2. The wall / margin dataset (ground truth, both architectures)

54 bracket rows total (30 mlp_s + 24 mlp_l), `kspec_results/tier2_dataset.csv` and
`kspec_results/arch/tier2_dataset.csv`. Censoring is explicit (bounds, never dropped).

**Coherent cells (r₁ ≤ −0.9), all optimizers, both architectures:** onset margin **0.025–0.05**
(= bracket grid floor), budget (death/onset) **1.12–1.19** where deaths observed. Optimizer- and
architecture-independent at bracket resolution.

**Noisy cells (mlp_s):** margins grow with noise — b32 0.10–0.15, b128 0.15, Adam b128 0.64
(deep interior), β0.99 b8 0.64 (the memory-residual outlier, 2× above the amplitude law).
b8 β0.9: onset (1.22, 1.30] with death at 1.30 — onset and death now *unresolved between*
(censored per 10.1(ii)); the razor-thin-budget anomaly claim rests on this censored row.

**Adam anomaly:** margin ≈ 0.22 at cv2h = 0 (b2048) — the preconditioner is an intrinsic noise
source invisible to fixed-θ curvature statistics (its full-batch plateau "weather" reaches loss
0.09 vs 3e-7 for SGDM). Flagged X-gap for fit v2.

**mlp_l caveat:** no bracket run on mlp_l has *ever* died (all rows death-censored), and the
3×-plateau-max excitation rule is effectively blind on its noisy cells (undisturbed plateau loss
is already large) — six of eight sub-edge margins remain censored even after the hotter
c = 1.6/1.9/2.3 pass. Margin-law replication on mlp_l is **undecidable at current resolution**
(4 finite (margin, cv2h) pairs, 2 at grid floor, and the two informative b32 seeds disagree 7.5×).
This is registered outcome-pending under ADDENDUM 10.1(iv), not a failure.

---

## 3. Architecture replication scorecard (mlp_l battery + Phase-2, DATA-ONLY)

⚠️ None of this block has been through the registered verification order yet
(blind gate → A2 table → constant comparison → Muon branches; ADDENDUM 10.1 closing).
The mlp_l blind gate (`kspec_results/arch/gates.json`) **has not been run** — it is still
legitimately blind.

| cell (b2048 unless noted) | mlp_s | mlp_l (A2 where applicable) | transfer? |
|---|---|---|---|
| SGD | (trivial gain) | **1.970 / 1.968**, gain 1.0000 | ✅ replicates |
| Nesterov | 2.002 (5 seeds) | **1.994 / 1.996**, gain 1.474 | ✅ replicates |
| Nesterov elevation ladder | 2.22 (b128) → 2.14 → 2.07 → 2.00 | 2.25 (b128) → 2.08 (b512) → 2.00 | ✅ structure replicates |
| SGDM β0.9 | 1.988 (5 seeds) | **1.584 / 1.758** (A2, 30k steps) | ❌ shortfall — see below |
| Adam β₁0.9 | 1.940, r₁ −0.96 | **0.276 / 0.312** (A2), r₁ −0.42 | ❌ not coherent there |
| Muon β0.95 | 1.34 (3 seeds), r₁ −0.60, nonstat | **1.879 / 1.801**, r₁ −0.999, nonstat | ↔ flipped (more coherent on mlp_l) |
| sub-edge block (b8/b32/β0.99/adam_b8) | all < 2 | all < 2 (0.014–0.39) | ✅ replicates |

**The A2 finding that needs a verdict:** tripling the budget (10k → 30k steps) moved SGDM κ_spec
only +1.5%/+3.7% and left Adam at ~0.3, while κ-drift fell to 0.005 (fully stationary) and gains
stayed on-formula. So the registered prediction (κ_spec ∈ [1.8, 2.2] once at-plateau) **failed**,
but the pre-stated escape branch ("still climbing at 30k → censor") is *also* unavailable — the
cells look plateaued and simply read low. Neither registered branch covers "plateaued at 1.6–1.8";
the analysis session must name this. Note the *bracket* side replicates fine: A2 SGDM s1 crossing
interval [3.786, 4.277] in the (1.15, 1.3] bin **contains 2(1+β) = 3.8** (per the 10.1(i) interval
readout, retiring the earlier "crosses at 4.0, elevated" first-pass claim). So the *wall* is where
the law says; it is the passive spectral reading that falls short on mlp_l for SGDM/Adam.
Most likely axis to check next: mlp_l's plateaus at this budget sit further below their walls
(r₁ −0.90/−0.96 vs −0.999 on mlp_s; Adam r₁ −0.42 vs −0.96), i.e. the *cells* are less coherent,
consistent with Tier-1's own domain condition rather than contradicting it — but that is a
tomorrow judgment, not a registered result.

---

## 4. Instrument-domain boundary findings (SETTLED unless noted)

These map where each instrument is valid — reported as findings, not failures.

1. **adam05 (β₁=0.5): the LTI instrument's validity boundary.** Registered fifth-threshold
   prediction FAILED exactly as the failure branch specified: κ_spec reads 0.5 while the cell is
   at its wall by ground truth. Domain condition: filter memory ≫ preconditioner adaptation
   timescale (β₁=0.9 satisfies it; 0.5 does not). `KSPEC_RESULTS.md` Boundary finding 1.
2. **Muon: instrument-invalid, same class (DATA-ONLY).** All 5 Muon cells (3 mlp_s seeds +
   2 mlp_l) fail the stationarity gate (κ-drift −0.32…−0.47), so per the pre-stated ADDENDUM 10
   branch the κ_spec readings (1.33–1.35 mlp_s; 1.80–1.88 mlp_l) are registered raw-frame
   instrument-invalid; brackets carry the wall claim alone. mlp_s brackets found **no onset up to
   c=1.5** (grid too cold — prediction "margin > 0" unresolved there); mlp_l onsets resolved
   (margins 0.099/0.223) but the excitation flags trip on a tiny base and need eyeballing.
   Unregistered observation worth a look: **GBS_med = 2.000–2.004 in all five Muon cells**, both
   architectures — the classic edge signature reading exactly 2 on the one optimizer whose
   spectral instrument is invalid.
3. **Adam's real filter ≠ textbook EMA, by +8%, systematically** (gain 0.0568–0.0574 across 5
   seeds vs ideal 0.0526). The loop is marginal against its *own measured* filter — why the
   formula-free instrument works where any formula would be wrong.
4. **Nesterov mid-batch overshoot (κ_spec 2.22 at b128) is a noise-elevated wall, not an
   artifact.** Elevation decays ~13/b (7.7% → 5.1% → 2.5% → 0.0%) at saturated coherence, and the
   bracket confirms the operative wall sits above the deterministic law. Replicates on mlp_l.
5. **b8 pooled frame does not converge** (held-out capture 0.51 at K=120): the unstable family at
   small batch is not low-dimensional. Operator estimators (i)/(ii) are invalid there by
   measurement; full-space replicas (iii) — always quoted as excess-over-flat-control — are the
   only valid MS estimator at b8.
6. **Instrument hygiene that bit us (permanent rules):** stride-2 aliases ω=π to DC (stride 1
   mandatory); checkpoints must be in the live phase (ring-down excluded); (iii) never quoted
   bare (injection floor); censored rows enter as bounds, never dropped.

---

## 5. Graveyard (short — what was tried and killed)

Full list: `SUMMARY.md` Part V + agent-era addenda. One line each:

- **Universal scalar = 2 at every plateau, any moment order** — measured dead (Tier-1 strong form + Reading A).
- **Reading A (κ_ms = 2 pins every plateau)** — dead; plateaus sit *inside* the wall with noise-dependent margin (Reading B stands).
- **GBS = 2 as universal (one-coordinate)** — parks below 2 at small batch for EVERY optimizer incl. SGD-at-low-lr; ceiling explained by the memory-edge min-law (2026-08-30, `analysis/MEMORY_EDGE_LAW.md`); the old "heavy-ball large-batch GBS ≠ 2" readings were float32-dead-run artifacts.
- **R = memory/τ_rot composite** — free-exponent regression killed it (τ_rot exponent ≈ 0); replaced by min(2(1+β), reach(batch)).
- **KKT/metastable "force-free phase"** — retracted by the park-vs-attractor test (regulated attractor; `SUMMARY.md` Parts III/IV carry the retraction header).
- **Fast/noise instruments** (γ-kick, AR-pole, point frozen-cocycle, burstiness, event rate, excursion tails, sharpening-suppression) — all killed; pattern: fast statistics carry noise info, not regime info.
- **Paper Eq-21 1D frame-blind MS law** — exact at the deterministic end (ratios 0.99–1.02), fails the interpolation band (0.4–1.4×, seed-inconsistent).
- **"Thin budget ∝ R" law** — failed as monotone law; single surviving anomaly now rests on a censored row.
- **Top-3-enriched frame** as the (i)/(ii) bias explanation — refuted; **τ_su** as the β0.99 memory-X — refuted.

---

## 6. Open queue (registered, in order)

1. **Run the mlp_l blind gate** (`kspec_gate.py` on arch cells — still blind, do this first).
2. **A2 verdict**: name the "plateaued but reads 1.6–1.8" branch; write the crossing intervals per 10.1(i).
3. **ADDENDUM 9/10.1(iv) constant comparison**: currently undecidable — needs hotter mlp_l brackets or a fluctuation-scaled excitation rule for noisy plateaus.
4. **Tier-2 refit v2** without the censored anchors (+ candidate new X's: memory residual for β0.99, preconditioner-noise for Adam; out-of-sample only).
5. **Muon follow-ups**: hotter mlp_s bracket grid (nothing excited at 1.5×); eyeball the mlp_l onset flags; layer-spectral frame = deferred branch (b); the GBS=2.00 observation (RESOLVED at b2048: real, health-checked, gold row; Muon small-batch ceiling does NOT fit C(B)/(1−0.95) in the raw frame — open with branch (b)).
6. **κ_ms single-construction recompute** across all cells (est-(i) with its measured +0.3 bias band; empirical onsets as truth) — FINAL TABLE RULE, ADDENDUM 5.
7. ~~Wide-grid collapse figure~~ — DONE 2026-08-30 (`analysis/collapse_*_gold.png`): registered criterion failed for every single coordinate; min-law stands.
8. Escape-rate reconciliation + the unidentified interior regulator (`SUMMARY.md` Part VI).

---

## 7. Full numeric tables

The consolidated per-cell tables (κ_spec ± CI, gain, r₁, κ_raw, flags for every mlp_s `L_*` and
mlp_l `A_*`/`A2_*` cell; κ_ms/γ₂; both tier2 datasets; Muon) live in the JSON/CSV under
`kspec_results/` and `kspec_results/arch/` — `*_kspec.json` per cell, `*_gamma.json` for κ_ms,
`tier2_dataset.{json,csv}`, `bracket.json` for raw bracket runs, `gates.json` (mlp_s only so far),
`trio_anchor.json`, `frame_audit.json`, `paperlaw.json`. Every number in this document traces to
one of those files or to a table in `KSPEC_RESULTS.md`.

---

## 8. Document index

| Document | What's in it |
|---|---|
| `KSPEC_RESULTS.md` | The flagship results doc: Headlines 1–3 (gate pass, strong-form refutation, wall table), five-optimizer table, boundary findings, wall dataset, Tier-2 verdict. **Predates the overnight Phase-2 data** — §3/§4.2/§6 above cover the gap. |
| `KSPEC_PREREG_ANNOTATIONS.md` | All 10 registered addenda: every prediction, gate, censoring rule, and interpretation branch, committed before its data. The audit trail. |
| `KSPEC_DESIGN.md` | Original κ_spec hypothesis, estimator spec, the blind gate criteria. |
| `SUMMARY.md` | The long arc: Parts I–VI (pre-κ_spec era, incl. retractions marked in place), Part VII (two-tier final picture). |
| `HANDOFF.md` | Live-thread state: open items, where each dataset lives, run/rerun instructions. |
| `LESSONS.md` | Methodology rules learned the hard way (blind gates, censoring, stride, ring-down...). |
| `P1_DESIGN.md` | Older phase-1 design (endogenous-predictor circularity note). |
| This file | The synthesis. Update it when the §6 queue items are adjudicated. |
