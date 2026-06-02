"""Curvature scan helpers for the central-flow failure-mode experiment.

Implements two scans used inside ProjectionTracker.post_step to characterize
how curvature varies in the neighborhood of an EoS oscillation:

(a) compute_curvature_segment — scans u^T H(w(α)) u along the line segment
    w(α) = α*w_t + (1-α)*w_{t+1} with a fixed eigvec u (typically u_mid =
    top eigvec at the segment midpoint). This is the Figure 29 diagnostic
    from Cohen et al. (Understanding Optimization with Central Flows, p.89).

(b) compute_curvature_along_u — scans u^T H(w_t + β*u) u along the top
    Hessian eigvec at a single iterate.

Both helpers snapshot the current parameter vector, run the scan with
in-place `.data.copy_` writes, and restore the original params in a
try/finally. Each interior HVP is freed before the next iteration so GPU
memory stays bounded across the scan.
"""

from __future__ import annotations

import numpy as np
import torch

from .measure import (
    param_vector,
    create_hessian_vector_product,
)


def set_params_inplace(net, theta_flat: torch.Tensor) -> None:
    """Copy a flat 1D tensor into net.parameters().data (trainable only).

    Uses `.data.copy_` so the Parameter object identity (and any
    optimizer.state[p] buffers keyed on it) is preserved.
    """
    offset = 0
    for p in net.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        p.data.copy_(theta_flat[offset:offset + n].view_as(p))
        offset += n
    if offset != theta_flat.numel():
        raise ValueError(
            f"set_params_inplace: flat tensor has {theta_flat.numel()} elements "
            f"but trainable params total {offset}."
        )


def _rayleigh_uHu_at(net, loss_fn, X_sub, Y_sub, u: torch.Tensor) -> float:
    """Compute u^T H(current net params) u on the fixed subset, via one HVP.

    Caller is responsible for setting net.parameters() to the desired w.
    """
    with torch.enable_grad():
        preds = net(X_sub).squeeze(dim=-1)
        loss = loss_fn(preds, Y_sub)
    hvp = create_hessian_vector_product(loss, net, retain_graph=False)
    try:
        Hu = hvp(u, retain_graph_override=False)
        s = torch.dot(u, Hu).item()
    finally:
        hvp.free_memory()
    return s


def compute_curvature_segment(
    net,
    loss_fn,
    X_sub: torch.Tensor,
    Y_sub: torch.Tensor,
    theta_t: torch.Tensor,
    theta_tp1: torch.Tensor,
    u_mid: torch.Tensor,
    alphas: np.ndarray,
) -> np.ndarray:
    """Return S[i] = u_mid^T H(α_i * θ_t + (1-α_i) * θ_{t+1}) u_mid for each α in alphas.

    The same fixed eigvec u_mid and the same fixed batch (X_sub, Y_sub) are
    used at every α — only the parameters change.

    Restores net params to the original values (whatever was set when called)
    via try/finally.
    """
    theta_save = param_vector(net).detach().clone()
    out = np.full(len(alphas), np.nan, dtype=np.float32)
    try:
        for i, a in enumerate(alphas):
            w_alpha = float(a) * theta_t + (1.0 - float(a)) * theta_tp1
            set_params_inplace(net, w_alpha)
            out[i] = _rayleigh_uHu_at(net, loss_fn, X_sub, Y_sub, u_mid)
    finally:
        set_params_inplace(net, theta_save)
    return out


def compute_curvature_along_u(
    net,
    loss_fn,
    X_sub: torch.Tensor,
    Y_sub: torch.Tensor,
    theta_t: torch.Tensor,
    u: torch.Tensor,
    betas: np.ndarray,
) -> np.ndarray:
    """Return S[i] = u^T H(θ_t + β_i * u) u for each β in betas.

    The eigvec u and batch (X_sub, Y_sub) are fixed across the scan.
    """
    theta_save = param_vector(net).detach().clone()
    out = np.full(len(betas), np.nan, dtype=np.float32)
    try:
        for i, b in enumerate(betas):
            w_beta = theta_t + float(b) * u
            set_params_inplace(net, w_beta)
            out[i] = _rayleigh_uHu_at(net, loss_fn, X_sub, Y_sub, u)
    finally:
        set_params_inplace(net, theta_save)
    return out
