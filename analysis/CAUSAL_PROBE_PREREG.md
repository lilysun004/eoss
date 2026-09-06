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

---
> **[VERDICT 2026-09-06 14:00, after data — CAUSAL_PROBE_RESULTS.md, results/kspec_causal/.]**
>
> **P1 (noise ablation):** kill NOT triggered overall, with one registered exception. Scoring note,
> on the record: P1a as written compared post-swap λ to pre-swap λ_B — a frame error (λ_B changes
> identity at the swap); both readings reported. Frame-consistent (vs pre-swap λ_full): SGD +34%,
> SGDM +38% by +6k (rising; SLOWER than the registered ≥30%@2k — strict P1a FAIL on rate, direction
> confirmed), Adam +2225% in the whitened frame (violent resume). **Muon: flat/−18% → NOT
> noise-maintained** (registered kill clause applies to Muon alone).
> **P2 (attractor prediction):** Adam landed at ratio **0.93**, Muon **1.05** of the pre-registered
> de-novo targets (band ±25%) — both PASS, stationary. SGDM took the registered violent branch:
> catapult divergence 16 steps post-swap (loss 6.6e-8 → 8.7e4) — the plateau is NOT globally
> attracting for heavy ball from ~11× above.
> **P3 (kick train):** registered branch "mixed/other" — reality exceeded both hypotheses for the
> SGD family: 32×A0 kick KILLED SGD (step 13203), 128×A0 killed SGDM (13835) and Nesterov (13814);
> 2–8×A0 kicks relaxed with λ_B within ±3.5%. **Adam and Muon survived all 20 kicks each incl.
> 5×128×A0** (λ̃ change ≤ +17% / ≤ ±3%).
>
> **Mechanism answer ("why is the small-batch point stable"):**
> - **SGD family (LTI):** a noise-maintained marginal state with a FINITE BASIN — weak local
>   restoring, sharpening resumes when noise is removed, and a hard instability cliff a few
>   fluctuation-widths along the top mode (sharp-edge picture wins over soft balance for this family;
>   resolves the audit's open question #3 in favor of "edge" — at the basin boundary, not in the
>   local λ response).
> - **Adam:** noise-maintained AND globally robust — the preconditioner absorbs 128× kicks and a 7×
>   curvature drop, relaxing to the numerically pre-registered endpoint (0.93). Constant-gap plateau
>   location + soft response everywhere probed.
> - **Muon:** NOT noise-maintained, yet strongly attracting (predicted from above, 1.05) and inert to
>   kicks — its plateau is enforced by the update normalization itself, outside the curvature-noise
>   mechanism entirely.
