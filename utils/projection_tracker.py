import torch
import numpy as np
from pathlib import Path

from .measure import (
    param_vector,
    grads_vector,
    gimme_random_subset_idx,
    compute_eigenvalues,
    EigenvectorCache,
    HessianVectorProduct,
    create_hessian_vector_product,
    param_length,
)
from utils.nets import CNN, ResNet, WideResNet, WideResNetNoBN


class ProjectionTracker:
    """
    Tracks the projection of the parameter vector theta_t onto dynamically-
    relevant directions at each training step within a user-specified window.

    Standard directions (all optimizers):
      1. E[g_t]     — full-batch gradient (random subset approximation)
      2. g_t        — current mini-batch gradient
      3. h_t        — raw parameter change Δθ = θ_{t+1} − θ_t
      4. u_from     — top eigenvector of full Hessian fixed at track_from (if fixed_u=True)
      5. w_t        — top eigenvector of the full (subset) Hessian (per-step)
      6. w_b_t      — top eigenvector of the batch Hessian (per-step)

    Extra directions (Adam / RMSProp only, i.e. when optimizer has a preconditioner):
      7. ũ_from     — top eigenvec of D^{-1/2}HD^{-1/2} fixed at track_from (if fixed_u=True)
      8. w_p_t      — top eigenvector of D^{-1/2} H D^{-1/2} (per-step)
      9. w_pb_t     — top eigenvector of D^{-1/2} H_B D^{-1/2} (per-step)

    For (8) and (9), the eigenvalue λ_max(D^{-1/2} H D^{-1/2}) is also recorded.

    fixed_u=True: compute top eigenvectors once at track_from, reuse for all steps.
    This tests Lily's hypothesis that the instability direction is stable across time.

    Usage in training loop:
        theta_before = tracker.pre_step(step_number, X_batch, Y_batch, batch_loss)
        optimizer.step()
        tracker.post_step(step_number, theta_before)
    """

    def __init__(self, net, X, Y, loss_fn, track_from, track_until,
                 save_dir, device, optimizer_wrapper=None, fixed_u: bool = False):
        self.net = net
        self.X = X
        self.Y = Y
        self.loss_fn = loss_fn
        self.track_from = track_from
        self.track_until = track_until
        self.save_dir = Path(save_dir)
        self.device = device
        self._optimizer_wrapper = optimizer_wrapper
        self._has_preconditioner = (
            optimizer_wrapper is not None and
            callable(getattr(optimizer_wrapper, 'get_preconditioner_inv_sqrt', None))
        )
        self._fixed_u = fixed_u

        # Fixed eigenvectors captured at track_from (only used when fixed_u=True)
        self._fixed_u_vec:        torch.Tensor | None = None
        self._fixed_u_precond_vec: torch.Tensor | None = None

        # Warm-start eigenvector caches
        self._eigvec_cache_full          = EigenvectorCache(max_eigenvectors=1)
        self._eigvec_cache_batch         = EigenvectorCache(max_eigenvectors=1)
        self._eigvec_cache_precond_full  = EigenvectorCache(max_eigenvectors=1)
        self._eigvec_cache_precond_batch = EigenvectorCache(max_eigenvectors=1)

        # Storage (ordering: g_full, g, h, w_fixed, w, wb, w_precond_fixed, w_precond, wb_precond)
        self._steps:               list[int]   = []
        self._proj_g_full:         list[float] = []
        self._proj_g:              list[float] = []
        self._proj_h:              list[float] = []
        self._proj_w_fixed:        list[float] = []
        self._proj_w:              list[float] = []
        self._proj_wb:             list[float] = []
        self._proj_w_precond_fixed: list[float] = []
        self._proj_w_precond:      list[float] = []
        self._proj_wb_precond:     list[float] = []
        self._lambda_precond:       list[float] = []
        self._lambda_precond_batch: list[float] = []

        # Temporary per-step state
        self._pending_step:      int | None          = None
        self._pending_theta_t:   torch.Tensor | None = None
        self._pending_g_t:       torch.Tensor | None = None
        self._pending_g_full:    torch.Tensor | None = None
        self._pending_w_t:       torch.Tensor | None = None
        self._pending_wb_t:      torch.Tensor | None = None
        self._pending_w_precond:   torch.Tensor | None = None
        self._pending_wb_precond:  torch.Tensor | None = None
        self._pending_lam_precond: float | None = None
        self._pending_lam_precond_batch: float | None = None

    def should_track(self, step: int) -> bool:
        return self.track_from <= step <= self.track_until

    # ------------------------------------------------------------------ #
    #  Subset selection                                                    #
    # ------------------------------------------------------------------ #

    def _get_subset_cap(self) -> int:
        cap = 4096
        if str(self.device).startswith('cuda'):
            try:
                total_memory = torch.cuda.get_device_properties(self.device).total_memory
            except Exception:
                total_memory = float('inf')
            if total_memory < 20 * 1024 ** 3:
                if isinstance(self.net, CNN):
                    cap = 2048 + 512
                elif isinstance(self.net, ResNet):
                    cap = 512
                elif isinstance(self.net, (WideResNet, WideResNetNoBN)):
                    cap = 1024
                elif self.net.__class__.__name__ == 'ViT':
                    cap = 2048
        return cap

    def _get_subset(self):
        cap = self._get_subset_cap()
        N = len(self.X)
        if N > cap:
            idx = gimme_random_subset_idx(N, cap)
            return self.X[idx], self.Y[idx]
        return self.X, self.Y

    # ------------------------------------------------------------------ #
    #  Standard power iteration (for Hessian eigenvectors)                #
    # ------------------------------------------------------------------ #

    def _power_iteration(self, loss, cache, max_iters=50, reltol=0.005):
        """Return (eigenvalue, eigenvector) for the top eigenvector of H."""
        _, w = compute_eigenvalues(
            loss, self.net,
            k=1,
            max_iterations=max_iters,
            reltol=reltol,
            eigenvector_cache=cache,
            return_eigenvectors=True,
            use_power_iteration=True,
        )
        # Retrieve the eigenvalue stored in cache by the power iteration
        lam = cache.eigenvalues[0] if cache.eigenvalues else float('nan')
        if torch.is_tensor(lam):
            lam = lam.item()
        return lam, w.detach().clone()

    # ------------------------------------------------------------------ #
    #  Preconditioned power iteration for D^{-1/2} H D^{-1/2}            #
    # ------------------------------------------------------------------ #

    def _precond_power_iteration(self, loss, cache, D_inv_sqrt,
                                  max_iters=50, reltol=0.005):
        """Return (eigenvalue, eigenvector) for the top eigenvector of D^{-1/2} H D^{-1/2}.

        Matvec: v → D^{-1/2} H (D^{-1/2} v)
        """
        hvp = create_hessian_vector_product(loss, self.net, retain_graph=True)
        n = param_length(self.net)
        d = D_inv_sqrt.to(self.device)

        try:
            # Warm start from cache
            if cache.eigenvectors:
                v = cache.eigenvectors[0].to(self.device).detach()
            else:
                v = torch.randn(n, device=self.device, dtype=d.dtype)

            with torch.no_grad():
                v = v / v.norm()

            eigenval = 0.0
            for _ in range(max_iters):
                # Preconditioned matvec: D^{-1/2} H D^{-1/2} v
                u = d * v
                Hu = hvp(u).detach()
                pHv = d * Hu

                with torch.no_grad():
                    rq = torch.dot(pHv, v) / torch.dot(v, v)
                    eigenval = rq

                    if rq.abs() < 1e-12:
                        break
                    residual = pHv - rq * v
                    if residual.norm() / rq.abs() < reltol:
                        break
                    v = pHv / pHv.norm()

            v = v.detach().clone()
            cache.store_eigenvector(v, eigenval)
            lam = eigenval.item() if torch.is_tensor(eigenval) else float(eigenval)
            return lam, v

        finally:
            hvp.free_memory()

    # ------------------------------------------------------------------ #
    #  pre_step / post_step                                                #
    # ------------------------------------------------------------------ #

    def pre_step(self, step: int, X_batch: torch.Tensor, Y_batch: torch.Tensor,
                 batch_loss: torch.Tensor) -> torch.Tensor:
        """Call after loss.backward(), before optimizer.step()."""

        # 1. Capture θ_t and g_t
        theta_t = param_vector(self.net).detach().clone()
        g_t = grads_vector(self.net).detach().clone()

        # Save original .grad so optimizer.step() is unaffected
        saved_grads = [
            p.grad.detach().clone() if p.grad is not None else None
            for p in self.net.parameters()
        ]

        # 2. Full-batch gradient (random subset)
        X_sub, Y_sub = self._get_subset()
        self.net.zero_grad()
        with torch.enable_grad():
            preds_sub = self.net(X_sub).squeeze(dim=-1)
            loss_sub = self.loss_fn(preds_sub, Y_sub)
            loss_sub.backward()
        g_full_t = grads_vector(self.net).detach().clone()

        # Restore mini-batch grads
        for p, g in zip(self.net.parameters(), saved_grads):
            p.grad = g

        # 3. w_t — top eigenvector of full Hessian (new graph)
        with torch.enable_grad():
            preds_sub2 = self.net(X_sub).squeeze(dim=-1)
            loss_sub2 = self.loss_fn(preds_sub2, Y_sub)
        _, w_t = self._power_iteration(loss_sub2, self._eigvec_cache_full)

        # Capture fixed full-Hessian eigenvector at the first tracked step
        if self._fixed_u and self._fixed_u_vec is None:
            self._fixed_u_vec = w_t.clone()

        # 4. w_b_t — top eigenvector of batch Hessian
        with torch.enable_grad():
            preds_b = self.net(X_batch).squeeze(dim=-1)
            loss_b = self.loss_fn(preds_b, Y_batch)
        _, w_b_t = self._power_iteration(loss_b, self._eigvec_cache_batch)

        # 5. Preconditioned eigenvectors (Adam / RMSProp only)
        w_precond = None
        w_b_precond = None
        lam_precond = float('nan')
        lam_precond_batch = float('nan')

        if self._has_preconditioner:
            D_inv_sqrt = self._optimizer_wrapper.get_preconditioner_inv_sqrt()
            if D_inv_sqrt is not None:
                # Full preconditioned Hessian eigenvector
                with torch.enable_grad():
                    preds_p = self.net(X_sub).squeeze(dim=-1)
                    loss_p = self.loss_fn(preds_p, Y_sub)
                lam_precond, w_precond = self._precond_power_iteration(
                    loss_p, self._eigvec_cache_precond_full, D_inv_sqrt)

                # Capture fixed preconditioned eigenvector at the first tracked step
                if self._fixed_u and self._fixed_u_precond_vec is None:
                    self._fixed_u_precond_vec = w_precond.clone()

                # Batch preconditioned Hessian eigenvector
                with torch.enable_grad():
                    preds_pb = self.net(X_batch).squeeze(dim=-1)
                    loss_pb = self.loss_fn(preds_pb, Y_batch)
                lam_precond_batch, w_b_precond = self._precond_power_iteration(
                    loss_pb, self._eigvec_cache_precond_batch, D_inv_sqrt)

        # Stash for post_step
        self._pending_step           = step
        self._pending_theta_t        = theta_t
        self._pending_g_t            = g_t
        self._pending_g_full         = g_full_t
        self._pending_w_t            = w_t
        self._pending_wb_t           = w_b_t
        self._pending_w_precond      = w_precond
        self._pending_wb_precond     = w_b_precond
        self._pending_lam_precond    = lam_precond
        self._pending_lam_precond_batch = lam_precond_batch

        return theta_t

    def post_step(self, step: int, theta_before: torch.Tensor):
        """Call after optimizer.step()."""
        assert self._pending_step == step, "post_step called for wrong step"

        theta_after = param_vector(self.net).detach()
        h_t = theta_after - theta_before
        theta_t = self._pending_theta_t

        dot = torch.dot

        self._steps.append(step)
        self._proj_g_full.append(dot(theta_t, self._pending_g_full).item())
        self._proj_g.append(dot(theta_t, self._pending_g_t).item())
        self._proj_h.append(dot(theta_t, h_t).item())

        # Fixed full-Hessian eigenvector projection (NaN if fixed_u=False)
        if self._fixed_u_vec is not None:
            self._proj_w_fixed.append(dot(theta_t, self._fixed_u_vec).item())
        else:
            self._proj_w_fixed.append(float('nan'))

        self._proj_w.append(dot(theta_t, self._pending_w_t).item())
        self._proj_wb.append(dot(theta_t, self._pending_wb_t).item())

        # Fixed preconditioned eigenvector projection (NaN if fixed_u=False or no preconditioner)
        if self._fixed_u_precond_vec is not None:
            self._proj_w_precond_fixed.append(dot(theta_t, self._fixed_u_precond_vec).item())
        else:
            self._proj_w_precond_fixed.append(float('nan'))

        if self._pending_w_precond is not None:
            self._proj_w_precond.append(dot(theta_t, self._pending_w_precond).item())
        else:
            self._proj_w_precond.append(float('nan'))

        if self._pending_wb_precond is not None:
            self._proj_wb_precond.append(dot(theta_t, self._pending_wb_precond).item())
        else:
            self._proj_wb_precond.append(float('nan'))

        self._lambda_precond.append(self._pending_lam_precond)
        self._lambda_precond_batch.append(self._pending_lam_precond_batch)

        # Clear stash
        self._pending_step = None
        self._pending_theta_t = None
        self._pending_g_t = None
        self._pending_g_full = None
        self._pending_w_t = None
        self._pending_wb_t = None
        self._pending_w_precond = None
        self._pending_wb_precond = None
        self._pending_lam_precond = None
        self._pending_lam_precond_batch = None

    def save(self):
        """Save all recorded projections to projections.npz in save_dir."""
        if not self._steps:
            print("ProjectionTracker: no data recorded, skipping save.")
            return

        n = len(self._steps)
        nan_arr = lambda: np.full(n, np.nan, dtype=np.float32)

        out_path = self.save_dir / 'projections.npz'
        np.savez(
            out_path,
            steps=np.array(self._steps, dtype=np.int64),
            proj_g_full=np.array(self._proj_g_full,         dtype=np.float32),
            proj_g=np.array(self._proj_g,                   dtype=np.float32),
            proj_h=np.array(self._proj_h,                   dtype=np.float32),
            proj_w_fixed=np.array(self._proj_w_fixed,       dtype=np.float32) if self._proj_w_fixed else nan_arr(),
            proj_w=np.array(self._proj_w,                   dtype=np.float32),
            proj_wb=np.array(self._proj_wb,                 dtype=np.float32),
            proj_w_precond_fixed=np.array(self._proj_w_precond_fixed, dtype=np.float32) if self._proj_w_precond_fixed else nan_arr(),
            proj_w_precond=np.array(self._proj_w_precond,   dtype=np.float32),
            proj_wb_precond=np.array(self._proj_wb_precond, dtype=np.float32),
            lambda_precond=np.array(self._lambda_precond,   dtype=np.float32),
            lambda_precond_batch=np.array(self._lambda_precond_batch, dtype=np.float32),
            track_from=np.int64(self.track_from),
            track_until=np.int64(self.track_until),
        )
        print(f"ProjectionTracker: saved {len(self._steps)} steps to {out_path}")
