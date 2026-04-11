import torch
import numpy as np
from pathlib import Path

from .measure import (
    param_vector,
    grads_vector,
    gimme_random_subset_idx,
    compute_eigenvalues,
    EigenvectorCache,
)
from utils.nets import CNN, ResNet, WideResNet, WideResNetNoBN


class ProjectionTracker:
    """
    Tracks the projection of the parameter vector theta_t onto five dynamically-
    relevant directions at each training step within a user-specified window.

    Directions tracked:
      1. E[g_t]  — full-batch gradient (approximated with a random subset)
      2. g_t     — current mini-batch gradient
      3. h_t     — raw parameter change Δθ = θ_{t+1} − θ_t  (filled in post-step)
      4. w_t     — top eigenvector of the full (approximated) Hessian
      5. w_b_t   — top eigenvector of the batch Hessian (current training mini-batch)

    Usage in training loop:
        # after loss.backward(), before optimizer.step():
        theta_before = tracker.pre_step(step_number, X_batch, Y_batch, batch_loss)

        optimizer.step()

        # after optimizer.step():
        tracker.post_step(step_number, theta_before)
    """

    def __init__(self, net, X, Y, loss_fn, track_from, track_until, save_dir, device):
        self.net = net
        self.X = X
        self.Y = Y
        self.loss_fn = loss_fn
        self.track_from = track_from
        self.track_until = track_until
        self.save_dir = Path(save_dir)
        self.device = device

        # Separate eigenvector caches for full-Hessian and batch-Hessian (warm starts)
        self._eigvec_cache_full = EigenvectorCache(max_eigenvectors=1)
        self._eigvec_cache_batch = EigenvectorCache(max_eigenvectors=1)

        # Storage arrays
        self._steps: list[int] = []
        self._proj_g_full: list[float] = []
        self._proj_g: list[float] = []
        self._proj_h: list[float] = []
        self._proj_w: list[float] = []
        self._proj_wb: list[float] = []

        # Temporary per-step state set in pre_step, consumed in post_step
        self._pending_theta_t: torch.Tensor | None = None
        self._pending_g_full: torch.Tensor | None = None
        self._pending_g_t: torch.Tensor | None = None
        self._pending_w_t: torch.Tensor | None = None
        self._pending_wb_t: torch.Tensor | None = None
        self._pending_step: int | None = None

    def should_track(self, step: int) -> bool:
        return self.track_from <= step <= self.track_until

    def _get_subset_cap(self) -> int:
        """Mirror the subset-size cap logic used by MeasurementRunner for lmax."""
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
        return cap

    def _get_subset(self):
        """Return a (possibly capped) random subset of the training data."""
        cap = self._get_subset_cap()
        N = len(self.X)
        if N > cap:
            idx = gimme_random_subset_idx(N, cap)
            return self.X[idx], self.Y[idx]
        return self.X, self.Y

    def pre_step(self, step: int, X_batch: torch.Tensor, Y_batch: torch.Tensor,
                 batch_loss: torch.Tensor) -> torch.Tensor:
        """
        Called after loss.backward(), before optimizer.step().

        Captures θ_t, g_t, E[g_t], w_t, w_b_t and stashes them for post_step.

        Returns theta_before (so the caller can pass it to post_step without
        re-computing it).
        """
        # 1. Capture θ_t and g_t before doing anything that could modify .grad
        theta_t = param_vector(self.net).detach().clone()
        g_t = grads_vector(self.net).detach().clone()

        # Save original .grad tensors so we can restore them after computing g_full_t
        saved_grads = [
            p.grad.detach().clone() if p.grad is not None else None
            for p in self.net.parameters()
        ]

        # 2. Compute g_full_t — full-batch gradient on a random subset
        X_sub, Y_sub = self._get_subset()
        self.net.zero_grad()
        with torch.enable_grad():
            preds_sub = self.net(X_sub).squeeze(dim=-1)
            loss_sub = self.loss_fn(preds_sub, Y_sub)
            loss_sub.backward()
        g_full_t = grads_vector(self.net).detach().clone()

        # Restore original mini-batch .grad values so optimizer.step() is unaffected
        for p, g in zip(self.net.parameters(), saved_grads):
            p.grad = g

        # 3. Compute w_t — top eigenvector of the full (subset) Hessian
        #    Use power iteration with warm starts: at EoS the eigenvector changes
        #    slowly, so warm-started power iteration converges in very few iters.
        with torch.enable_grad():
            preds_sub2 = self.net(X_sub).squeeze(dim=-1)
            loss_sub2 = self.loss_fn(preds_sub2, Y_sub)
        _, w_t = compute_eigenvalues(
            loss_sub2, self.net,
            k=1,
            max_iterations=50,
            reltol=0.005,
            eigenvector_cache=self._eigvec_cache_full,
            return_eigenvectors=True,
            use_power_iteration=True,
        )
        w_t = w_t.detach().clone()

        # 4. Compute w_b_t — top eigenvector of the batch Hessian
        #    Recompute the batch loss to get a fresh computation graph
        with torch.enable_grad():
            preds_b = self.net(X_batch).squeeze(dim=-1)
            loss_b = self.loss_fn(preds_b, Y_batch)
        _, w_b_t = compute_eigenvalues(
            loss_b, self.net,
            k=1,
            max_iterations=50,
            reltol=0.005,
            eigenvector_cache=self._eigvec_cache_batch,
            return_eigenvectors=True,
            use_power_iteration=True,
        )
        self._eigvec_cache_batch.store_eigenvector(w_b_t.detach())
        w_b_t = w_b_t.detach().clone()

        # Stash for post_step
        self._pending_step = step
        self._pending_theta_t = theta_t
        self._pending_g_t = g_t
        self._pending_g_full = g_full_t
        self._pending_w_t = w_t
        self._pending_wb_t = w_b_t

        return theta_t

    def post_step(self, step: int, theta_before: torch.Tensor):
        """
        Called after optimizer.step().

        Computes h_t = Δθ = θ_{t+1} − θ_t and records all five projections.
        """
        assert self._pending_step == step, "post_step called for wrong step"

        theta_after = param_vector(self.net).detach()
        h_t = theta_after - theta_before

        theta_t = self._pending_theta_t

        self._steps.append(step)
        self._proj_g_full.append(torch.dot(theta_t, self._pending_g_full).item())
        self._proj_g.append(torch.dot(theta_t, self._pending_g_t).item())
        self._proj_h.append(torch.dot(theta_t, h_t).item())
        self._proj_w.append(torch.dot(theta_t, self._pending_w_t).item())
        self._proj_wb.append(torch.dot(theta_t, self._pending_wb_t).item())

        # Clear stash
        self._pending_step = None
        self._pending_theta_t = None
        self._pending_g_t = None
        self._pending_g_full = None
        self._pending_w_t = None
        self._pending_wb_t = None

    def save(self):
        """Save all recorded projections to projections.npz in save_dir."""
        if not self._steps:
            print("ProjectionTracker: no data recorded, skipping save.")
            return

        out_path = self.save_dir / 'projections.npz'
        np.savez(
            out_path,
            steps=np.array(self._steps, dtype=np.int64),
            proj_g_full=np.array(self._proj_g_full, dtype=np.float32),
            proj_g=np.array(self._proj_g, dtype=np.float32),
            proj_h=np.array(self._proj_h, dtype=np.float32),
            proj_w=np.array(self._proj_w, dtype=np.float32),
            proj_wb=np.array(self._proj_wb, dtype=np.float32),
            track_from=np.int64(self.track_from),
            track_until=np.int64(self.track_until),
        )
        print(f"ProjectionTracker: saved {len(self._steps)} steps to {out_path}")
