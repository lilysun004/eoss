"""
Lyapunov-style edge-of-stability tracker via per-step JVPs.

Setup (general optimizer as discrete dynamical system)
------------------------------------------------------
State:  x_t = (θ_t, internal_state_t)
Step:   x_{t+1} = F(x_t; z_t)   where z_t is the mini-batch

Linearized perturbation propagation (same batch for both copies):
  δx_{t+1} ≈ J_t δx_t,   J_t = ∂F/∂x |_{(x_t, z_t)}

Per-step log growth (parameter-space norm):
  ell_t = log ‖δθ_{t+1}‖  (with ‖δθ_t‖ = 1 after renorm)

Pattern B — EMA at multiple time constants α (τ_eff ≈ 1/α):
  m_t(α) = (1-α) m_{t-1}(α) + α ell_t
  g_τ(t) = exp( m_t(α) )

Edge condition (optimizer-agnostic):
  g_τ ≈ 1   ⟺   ell on average ≈ 0

Why per-step JVP instead of periodic power iteration
-----------------------------------------------------
Power iteration on J_b = I - η·D·H_b has a trivial eigenvalue-1 null space
(any direction in null(H_b) gives J·v = v).  For mini-batch with batch << params
this null space is huge, so the iteration converges to ρ ≈ 1 always in the
stable regime — useless as a metric.

With the Lyapunov approach, stochastic batches rotate the null space each step.
A direction stuck at ell = 0 in one step is pushed by the next batch.
The EMA of ell therefore reflects genuine stochastic growth rates, not the
null-space artifact.

Implementation: parameter-only JVP with frozen optimizer state
--------------------------------------------------------------
For SGD (P = I):
  δθ' = δθ - η H_b δθ

For Adam (frozen diagonal preconditioner P_i = 1/(√v̂_i + ε)):
  δθ' = δθ - η P ⊙ (H_b δθ)

(The optimizer-state block of J — ∂m'/∂m = β₁, ∂v'/∂v = β₂ — contracts the
state perturbation each step.  Tracking it would give a slightly more accurate
estimate but requires maintaining extra perturbation buffers.  The parameter-only
projection captures the dominant parameter-space instability.)
"""

import torch
import math


# ---------------------------------------------------------------------------
# Core: one JVP step
# ---------------------------------------------------------------------------

def _compute_jvp(net, X_batch, Y_batch, loss_fn, optimizer_wrapper, dx):
    """Compute δθ' = J_t · δθ  (parameter-only, optimizer state frozen).

    Args:
        dx: flat parameter-space perturbation (detached, ‖dx‖ = 1 expected)

    Returns:
        dx_new (detached flat tensor): J_t · dx, NOT yet renormalised
    """
    lr = optimizer_wrapper.lr
    P  = optimizer_wrapper.get_diagonal_preconditioner()   # None → P = I

    params = list(net.parameters())

    net.zero_grad()
    preds = net(X_batch).squeeze(-1)
    loss  = loss_fn(preds, Y_batch)

    grads      = torch.autograd.grad(loss, params, create_graph=True)
    grads_flat = torch.cat([g.contiguous().view(-1) for g in grads])

    gv         = (grads_flat * dx).sum()
    hvp_grads  = torch.autograd.grad(gv, params)
    Hv         = torch.cat([g.contiguous().view(-1) for g in hvp_grads]).detach()

    del grads_flat, grads, loss, preds

    dx_new = dx - lr * (P * Hv if P is not None else Hv)
    return dx_new.detach()


# ---------------------------------------------------------------------------
# JacobianTracker: maintains perturbation vector + multi-τ EMAs
# ---------------------------------------------------------------------------

class JacobianTracker:
    """Per-step perturbation propagation with EMA log-growth tracking.

    Usage (inside training loop, before the optimizer step):
        metrics = tracker.step(net, X_batch, Y_batch, loss_fn, optimizer)
        # metrics: {'ell': float, 'rho_1': float, 'rho_10': float, ...}

    The edge condition is g_τ ≡ rho_τ ≈ 1  (ell ≈ 0 on average).
    Stable:    rho_τ < 1   Unstable: rho_τ > 1.

    Args:
        net:  nn.Module (used to get device and param count at init)
        taus: sequence of effective time constants τ; α = 1/τ
              τ = 1   → instantaneous (just exp(ell_t))
              τ = 10  → ~10-step EMA
              τ = 100 → ~100-step EMA
    """

    def __init__(self, net, taus=(1, 10, 100)):
        self.taus   = tuple(taus)
        self.alphas = tuple(1.0 / t for t in taus)
        self._emas  = {a: 0.0 for a in self.alphas}
        self._ready = False

        device      = next(net.parameters()).device
        n           = sum(p.numel() for p in net.parameters())
        v           = torch.randn(n, device=device)
        self.dx     = (v / v.norm()).detach()

    def step(self, net, X_batch, Y_batch, loss_fn, optimizer_wrapper):
        """Propagate perturbation one step, update EMAs, renormalise.

        Returns dict with keys: 'ell', and 'rho_{tau}' for each tau.
        """
        dx_new   = _compute_jvp(net, X_batch, Y_batch, loss_fn, optimizer_wrapper, self.dx)
        norm_new = dx_new.norm().item()

        ell = math.log(norm_new) if norm_new > 1e-30 else -30.0

        # Initialise EMAs on first call, then update
        for a in self.alphas:
            if not self._ready:
                self._emas[a] = ell
            else:
                self._emas[a] = (1.0 - a) * self._emas[a] + a * ell
        self._ready = True

        # Renormalise perturbation to keep numerics stable
        self.dx = (dx_new / norm_new).detach() if norm_new > 1e-30 else self.dx

        out = {'ell': ell}
        for tau, a in zip(self.taus, self.alphas):
            out[f'rho_{tau}'] = math.exp(self._emas[a])
        return out
