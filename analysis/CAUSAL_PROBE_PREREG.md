# CAUSAL_PROBE_PREREG.md — what maintains the small-batch plateau? (registered 2026-09-06, BEFORE runs)

**Question.** At the small-batch ceiling ("stable point"), is stationarity maintained by (E) an
instability/self-stabilization mechanism (edge) or (S) a soft balance of sharpening drive vs noise-induced
flattening — measured causally, optimizer-agnostic (same protocol for SGD/SGDM/Nesterov/Adam/Muon).

## Probes and REGISTERED predictions

**P1 — noise ablation (batch swap UP, b8 → b2048 at step 12000, lr unchanged).** Cells: sgd(0.01),
sgdm β0.9(0.002), adam(0.001), muon mom0.95(0.001); 18000 steps.
- P1a (noise-maintenance, kill test): within 2000 post-swap steps, λ_B rises ≥ 30% above its
  pre-swap plateau median and keeps rising (≥ +10%/2k steps) for EVERY optimizer. If λ_B stays within
  ±10% for 6000 steps in ≥ 2 optimizers → the plateau is NOT noise-maintained (kill for (S) and for the
  noise-gap mechanism reading).
- P1b (direction): post-swap κ moves TOWARD that optimizer's gold coherent-edge value; no prediction on
  reaching it within budget.

**P2 — reverse ablation (batch swap DOWN, b2048 → b8 at step 5000, lr unchanged).** Cells: sgdm(0.0065),
adam(0.001), muon(0.001); 17000 steps. (sgdm swaps before its float32 death at ~6k; the post-swap
transient may be violent — divergence is an admissible, informative outcome, flagged not censored.)
- P2 (predictive/attractor): λ_B falls and the post-swap stationary κ lands within ±25% of the DE-NOVO
  b8 ceiling at that lr's scale — predicted endpoints (κ = lr·λ): sgdm → κ_full(b8-scale) + 2(1−β)·(les
  measured de-novo: 0.34·(0.0065/0.002) lr-scaling NOT assumed; registered comparison is λ_B, i.e.
  λ endpoint within ±25% of λ = 0.343/0.002 = 172 for sgdm, 5.08/0.001 = 5080 (whitened) for adam,
  0.259/0.001 = 259 for muon). Path-independence = same stable point reached from above.

**P3 — kick train (restoring force + mechanism discriminator).** Cells: sgd/sgdm/nest/adam/muon at
their gold b8 lrs; 24000 steps; from step 12000, every 600 steps, θ += a·u_B with a cycling through
{2, 8, 32, 128} × A0 and alternating sign, A0 = median|dxu| over steps 8000–12000 (per-run, measured
online; amplitudes logged).
- P3a (attractor): the kicked coordinate relaxes (|x_u| e-folds within 600 steps) for both signs, all
  optimizers — replicates the park-test regulated attractor at every optimizer.
- P3b (E vs S discriminator, the main reading): (E) predicts kicks with a ≥ 32·A0 are followed within
  300 steps by a λ_B DROP ≥ 10% relative to the 600-step pre-kick median (self-stabilization event),
  scaling with a, while a ≤ 8·A0 kicks leave λ_B within ±5%; (S) predicts λ_B within ±5% after ALL
  kick sizes (kicks relax through the loss without touching curvature). Mixed/other → report as-is.

## Rules
Standing rules apply (raw primitives, health mask, censored-not-dropped, no doc edits by the pipeline,
data-only assembly). lr/batch combos: post-swap combos reuse gold-validated (optimizer, batch, lr) where
they exist; sgdm b2048@0.002 and swaps flagged as unvalidated-but-conservative. Analysis windows and
thresholds above are FROZEN; no other statistic will be consulted for the P1/P2/P3 verdicts.
