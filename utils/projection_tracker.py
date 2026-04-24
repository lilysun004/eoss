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
    _run_lobpcg_with_operator,
    param_length,
)
from utils.nets import CNN, ResNet, WideResNet, WideResNetNoBN


TOP_K = 5


class ProjectionTracker:
    """
    Tracks the projection of the parameter vector theta_t onto the top-5 eigenvectors
    of various Hessian-like operators at each training step within a user-specified
    window. Also records top-5 cosine similarity between consecutive tracked steps
    (alignment of the eigenspace across steps).

    Quantities (all shaped [n_tracked_steps, 5] unless noted):

    Always computed:
      - proj_g_full [n]         : ⟨θ_t, E[g_t]⟩ (scalar per step)
      - proj_g      [n]         : ⟨θ_t, g_t⟩ (scalar per step)
      - proj_h      [n]         : ⟨θ_t, Δθ⟩ (scalar per step)
      - proj_w_top5 [n, 5]      : ⟨θ_t, w_k^full(t)⟩ for k=1..5, per-step full-H eigvecs
      - proj_wb_top5 [n, 5]     : ⟨θ_t, w_k^batch(t)⟩ for k=1..5, per-step batch-H eigvecs
      - proj_w_fixed_top5 [n, 5]: ⟨θ_t, w_k^fixed⟩, fixed at the first tracked step
      - cos_sim_full_top5 [n, 5]: |⟨w_k^full(t-1), w_k^full(t)⟩|; first row all NaN
      - lambda_top5 [n, 5]      : top-5 eigenvalues of full (subset) Hessian

    Preconditioned quantities (Adam / RMSProp only; NaN otherwise):
      - proj_w_precond_top5      [n, 5]
      - proj_wb_precond_top5     [n, 5]
      - proj_w_precond_fixed_top5[n, 5]
      - cos_sim_precond_top5     [n, 5]
      - lambda_precond_top5      [n, 5]

    The `fixed_u` constructor flag is retained for CLI backward compatibility but is
    now a no-op — fixed top-5 eigenvectors are always captured at the first tracked
    step.

    Usage in training loop:
        theta_before = tracker.pre_step(step_number, X_batch, Y_batch, batch_loss)
        optimizer.step()
        tracker.post_step(step_number, theta_before)
    """

    def __init__(self, net, X, Y, loss_fn, track_from, track_until,
                 save_dir, device, optimizer_wrapper=None, fixed_u: bool = False,
                 save_every: int = 100, track_stride: int = 1,
                 lobpcg_max_iters: int = 20, lobpcg_reltol: float = 0.02):
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
        # fixed_u accepted for CLI backward compat but now a no-op (always fixed).
        self._fixed_u = fixed_u
        # Incremental save cadence: call save() every `save_every` tracked steps so
        # partial data survives wall-clock kills / preemption. 0 disables.
        self._save_every = int(save_every)
        # Stride inside the tracking window — only every Nth step is actually
        # tracked (saves most of the work since eigenvectors drift slowly).
        self._track_stride = max(1, int(track_stride))
        self._lobpcg_max_iters = int(lobpcg_max_iters)
        self._lobpcg_reltol = float(lobpcg_reltol)
        # Log once per run when the batch-H path is elided (batch_size >= dataset).
        self._batch_h_skip_logged = False

        # Fixed top-5 eigenvectors captured at first tracked step (always).
        self._fixed_top5_full:    torch.Tensor | None = None  # [n_params, 5]
        self._fixed_top5_precond: torch.Tensor | None = None  # [n_params, 5]

        # Previous tracked step's top-5 eigenvectors for cosine similarity.
        self._prev_w_top5_full:    torch.Tensor | None = None
        self._prev_w_top5_precond: torch.Tensor | None = None

        # Warm-start eigenvector caches (top-5 each).
        self._eigvec_cache_full          = EigenvectorCache(max_eigenvectors=TOP_K)
        self._eigvec_cache_batch         = EigenvectorCache(max_eigenvectors=TOP_K)
        self._eigvec_cache_precond_full  = EigenvectorCache(max_eigenvectors=TOP_K)
        self._eigvec_cache_precond_batch = EigenvectorCache(max_eigenvectors=TOP_K)

        # Fixed full-Hessian subset (drawn once, then reused across tracked steps so
        # warm-started LOBPCG actually benefits from step-to-step Hessian similarity).
        self._fixed_subset_X: torch.Tensor | None = None
        self._fixed_subset_Y: torch.Tensor | None = None

        # Per-tracked-step storage. Scalar lists for the simple quantities; lists of
        # length-5 arrays for the top-5 quantities (stacked to 2D on save()).
        self._steps:                       list[int]       = []
        self._proj_g_full:                 list[float]     = []
        self._proj_g:                      list[float]     = []
        self._proj_h:                      list[float]     = []
        self._proj_w_top5:                 list[np.ndarray] = []
        self._proj_wb_top5:                list[np.ndarray] = []
        self._proj_w_fixed_top5:           list[np.ndarray] = []
        self._cos_sim_full_top5:           list[np.ndarray] = []
        self._lambda_top5:                 list[np.ndarray] = []
        self._proj_w_precond_top5:         list[np.ndarray] = []
        self._proj_wb_precond_top5:        list[np.ndarray] = []
        self._proj_w_precond_fixed_top5:   list[np.ndarray] = []
        self._cos_sim_precond_top5:        list[np.ndarray] = []
        self._lambda_precond_top5:         list[np.ndarray] = []

        # Temporary per-step state (populated in pre_step, consumed in post_step)
        self._pending_step:              int | None = None
        self._pending_theta_t:           torch.Tensor | None = None
        self._pending_g_t:               torch.Tensor | None = None
        self._pending_g_full:            torch.Tensor | None = None
        self._pending_w_top5:            torch.Tensor | None = None  # [n_params, 5]
        self._pending_wb_top5:           torch.Tensor | None = None
        self._pending_w_precond_top5:    torch.Tensor | None = None
        self._pending_wb_precond_top5:   torch.Tensor | None = None
        self._pending_cos_full:          np.ndarray | None = None
        self._pending_cos_precond:       np.ndarray | None = None
        self._pending_lambda_top5:       np.ndarray | None = None
        self._pending_lambda_precond_top5: np.ndarray | None = None

    def should_track(self, step: int) -> bool:
        if not (self.track_from <= step <= self.track_until):
            return False
        return (step - self.track_from) % self._track_stride == 0

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
        if self._fixed_subset_X is None:
            cap = self._get_subset_cap()
            N = len(self.X)
            if N > cap:
                idx = gimme_random_subset_idx(N, cap)
                self._fixed_subset_X = self.X[idx]
                self._fixed_subset_Y = self.Y[idx]
            else:
                self._fixed_subset_X = self.X
                self._fixed_subset_Y = self.Y
        return self._fixed_subset_X, self._fixed_subset_Y

    # ------------------------------------------------------------------ #
    #  Top-5 eigenvector helpers                                           #
    # ------------------------------------------------------------------ #

    def _top5_eigenvectors(self, loss, cache, max_iters=None, reltol=None, grads=None):
        """Return (eigenvalues [5], eigenvectors [n_params, 5]) for the full Hessian,
        warm-started from cache. Eigenvalues returned in descending order by LOBPCG."""
        if max_iters is None:
            max_iters = self._lobpcg_max_iters
        if reltol is None:
            reltol = self._lobpcg_reltol
        eigvals, eigvecs = compute_eigenvalues(
            loss, self.net,
            k=TOP_K,
            max_iterations=max_iters,
            reltol=reltol,
            eigenvector_cache=cache,
            return_eigenvectors=True,
            use_power_iteration=False,
            grads=grads,
        )
        # Sort descending (LOBPCG returns ascending or descending depending on flavor;
        # _run_lobpcg_with_operator uses _eigh_ascending which sorts descending).
        # Be defensive: always enforce descending here.
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        return eigvals.detach(), eigvecs.detach().clone()

    def _top5_precond_eigenvectors(self, loss, cache, D_inv_sqrt,
                                   max_iters=None, reltol=None):
        """Return top-5 eigenvalues/vectors of D^{-1/2} H D^{-1/2} via LOBPCG."""
        if max_iters is None:
            max_iters = self._lobpcg_max_iters
        if reltol is None:
            reltol = self._lobpcg_reltol
        hvp = create_hessian_vector_product(loss, self.net, retain_graph=True)
        d = D_inv_sqrt.to(self.device)

        def precond_operator(v):
            # v may be [n_params, k]; HVP accepts a flat vector, so process per-column.
            if v.ndim == 1:
                u = d * v
                return d * hvp(u).detach()
            out = torch.empty_like(v)
            for j in range(v.shape[1]):
                u_j = d * v[:, j]
                out[:, j] = d * hvp(u_j).detach()
            return out

        try:
            eigvals, eigvecs = _run_lobpcg_with_operator(
                precond_operator,
                self.net,
                k=TOP_K,
                max_iterations=max_iters,
                reltol=reltol,
                init_vectors=None,
                eigenvector_cache=cache,
                return_eigenvectors=True,
            )
        finally:
            hvp.free_memory()

        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        return eigvals.detach(), eigvecs.detach().clone()

    # ------------------------------------------------------------------ #
    #  pre_step / post_step                                                #
    # ------------------------------------------------------------------ #

    def pre_step(self, step: int, X_batch: torch.Tensor, Y_batch: torch.Tensor,
                 batch_loss: torch.Tensor) -> torch.Tensor:
        """Call after loss.backward(), before optimizer.step()."""

        # 1. Capture θ_t and g_t
        theta_t = param_vector(self.net).detach().clone()
        g_t = grads_vector(self.net).detach().clone()

        # 2+3. Single forward pass on the fixed subset: compute g_full_t AND reuse the
        #      same graph + grads for top-5 LOBPCG (saves one forward + one
        #      autograd.grad call per tracked step).
        X_sub, Y_sub = self._get_subset()
        with torch.enable_grad():
            preds_sub = self.net(X_sub).squeeze(dim=-1)
            loss_sub = self.loss_fn(preds_sub, Y_sub)
            params_list = list(self.net.parameters())
            grads_sub = torch.autograd.grad(loss_sub, params_list, create_graph=True)
        g_full_t = torch.cat([g.detach().flatten() for g in grads_sub])

        lam_top5, w_top5 = self._top5_eigenvectors(
            loss_sub, self._eigvec_cache_full, grads=grads_sub,
        )  # [5], [n_params, 5]

        # Capture fixed full-Hessian top-5 at the first tracked step (always).
        if self._fixed_top5_full is None:
            self._fixed_top5_full = w_top5.clone()

        # Cosine similarity with previous tracked step's top-5 (|·| absorbs ±sign).
        if self._prev_w_top5_full is None:
            cos_full = np.full(TOP_K, np.nan, dtype=np.float32)
        else:
            cos_full = np.array([
                abs(torch.dot(self._prev_w_top5_full[:, k], w_top5[:, k])).item()
                for k in range(TOP_K)
            ], dtype=np.float32)
        self._prev_w_top5_full = w_top5.clone()

        # 4. Top-5 of batch Hessian. When the training batch covers the full
        # dataset (e.g. full-batch training at b=|X|), batch-H == full-H — reuse
        # the already-computed full-H top-5 and skip a second (very expensive)
        # LOBPCG call.
        skip_batch_h = X_batch.shape[0] >= len(self.X)
        if skip_batch_h:
            if not self._batch_h_skip_logged:
                print("ProjectionTracker: batch-H path skipped "
                      f"(batch {X_batch.shape[0]} >= dataset {len(self.X)}); "
                      "reusing full-H top-5.")
                self._batch_h_skip_logged = True
            wb_top5 = w_top5
        else:
            with torch.enable_grad():
                preds_b = self.net(X_batch).squeeze(dim=-1)
                loss_b = self.loss_fn(preds_b, Y_batch)
            _, wb_top5 = self._top5_eigenvectors(loss_b, self._eigvec_cache_batch)

        # 5. Preconditioned top-5 (Adam / RMSProp only)
        w_precond_top5 = None
        wb_precond_top5 = None
        lam_precond_top5 = np.full(TOP_K, np.nan, dtype=np.float32)
        cos_precond = np.full(TOP_K, np.nan, dtype=np.float32)

        if self._has_preconditioner:
            D_inv_sqrt = self._optimizer_wrapper.get_preconditioner_inv_sqrt()
            if D_inv_sqrt is not None:
                # Full preconditioned Hessian top-5
                with torch.enable_grad():
                    preds_p = self.net(X_sub).squeeze(dim=-1)
                    loss_p = self.loss_fn(preds_p, Y_sub)
                lam_p, w_precond_top5 = self._top5_precond_eigenvectors(
                    loss_p, self._eigvec_cache_precond_full, D_inv_sqrt)
                lam_precond_top5 = lam_p.cpu().numpy().astype(np.float32)

                if self._fixed_top5_precond is None:
                    self._fixed_top5_precond = w_precond_top5.clone()

                if self._prev_w_top5_precond is None:
                    cos_precond = np.full(TOP_K, np.nan, dtype=np.float32)
                else:
                    cos_precond = np.array([
                        abs(torch.dot(
                            self._prev_w_top5_precond[:, k],
                            w_precond_top5[:, k]
                        )).item()
                        for k in range(TOP_K)
                    ], dtype=np.float32)
                self._prev_w_top5_precond = w_precond_top5.clone()

                # Batch preconditioned Hessian top-5 (skip when batch covers
                # full dataset — same reasoning as non-precond path above).
                if skip_batch_h:
                    wb_precond_top5 = w_precond_top5
                else:
                    with torch.enable_grad():
                        preds_pb = self.net(X_batch).squeeze(dim=-1)
                        loss_pb = self.loss_fn(preds_pb, Y_batch)
                    _, wb_precond_top5 = self._top5_precond_eigenvectors(
                        loss_pb, self._eigvec_cache_precond_batch, D_inv_sqrt)

        # Stash for post_step
        self._pending_step                 = step
        self._pending_theta_t              = theta_t
        self._pending_g_t                  = g_t
        self._pending_g_full               = g_full_t
        self._pending_w_top5               = w_top5
        self._pending_wb_top5              = wb_top5
        self._pending_w_precond_top5       = w_precond_top5
        self._pending_wb_precond_top5      = wb_precond_top5
        self._pending_cos_full             = cos_full
        self._pending_cos_precond          = cos_precond
        self._pending_lambda_top5          = lam_top5.cpu().numpy().astype(np.float32)
        self._pending_lambda_precond_top5  = lam_precond_top5

        return theta_t

    def post_step(self, step: int, theta_before: torch.Tensor):
        """Call after optimizer.step()."""
        assert self._pending_step == step, "post_step called for wrong step"

        theta_after = param_vector(self.net).detach()
        h_t = theta_after - theta_before
        theta_t = self._pending_theta_t

        dot = torch.dot

        def _projections(theta, W):
            # W: [n_params, 5] → return np array [5] of ⟨theta, W[:, k]⟩
            if W is None:
                return np.full(TOP_K, np.nan, dtype=np.float32)
            return np.array(
                [dot(theta, W[:, k]).item() for k in range(TOP_K)],
                dtype=np.float32,
            )

        self._steps.append(step)
        self._proj_g_full.append(dot(theta_t, self._pending_g_full).item())
        self._proj_g.append(dot(theta_t, self._pending_g_t).item())
        self._proj_h.append(dot(theta_t, h_t).item())

        self._proj_w_top5.append(_projections(theta_t, self._pending_w_top5))
        self._proj_wb_top5.append(_projections(theta_t, self._pending_wb_top5))
        self._proj_w_fixed_top5.append(_projections(theta_t, self._fixed_top5_full))
        self._cos_sim_full_top5.append(self._pending_cos_full)
        self._lambda_top5.append(self._pending_lambda_top5)

        self._proj_w_precond_top5.append(_projections(theta_t, self._pending_w_precond_top5))
        self._proj_wb_precond_top5.append(_projections(theta_t, self._pending_wb_precond_top5))
        self._proj_w_precond_fixed_top5.append(_projections(theta_t, self._fixed_top5_precond))
        self._cos_sim_precond_top5.append(self._pending_cos_precond)
        self._lambda_precond_top5.append(self._pending_lambda_precond_top5)

        # Clear stash
        self._pending_step = None
        self._pending_theta_t = None
        self._pending_g_t = None
        self._pending_g_full = None
        self._pending_w_top5 = None
        self._pending_wb_top5 = None
        self._pending_w_precond_top5 = None
        self._pending_wb_precond_top5 = None
        self._pending_cos_full = None
        self._pending_cos_precond = None
        self._pending_lambda_top5 = None
        self._pending_lambda_precond_top5 = None

        if self._save_every and len(self._steps) % self._save_every == 0:
            self.save()

    def save(self):
        """Save all recorded projections to projections.npz in save_dir."""
        if not self._steps:
            print("ProjectionTracker: no data recorded, skipping save.")
            return

        def stack(rows):
            return np.stack(rows, axis=0).astype(np.float32)

        out_path = self.save_dir / 'projections.npz'
        # np.savez auto-appends .npz; name the temp file so that the appended
        # suffix lands us at 'projections.tmp.npz', which we then atomic-rename.
        tmp_path = self.save_dir / 'projections.tmp.npz'
        # Write to temp path then atomic-rename so a mid-write kill can't corrupt
        # the existing snapshot.
        np.savez(
            tmp_path,
            steps=np.array(self._steps, dtype=np.int64),
            proj_g_full=np.array(self._proj_g_full, dtype=np.float32),
            proj_g=np.array(self._proj_g, dtype=np.float32),
            proj_h=np.array(self._proj_h, dtype=np.float32),
            proj_w_top5=stack(self._proj_w_top5),
            proj_wb_top5=stack(self._proj_wb_top5),
            proj_w_fixed_top5=stack(self._proj_w_fixed_top5),
            cos_sim_full_top5=stack(self._cos_sim_full_top5),
            lambda_top5=stack(self._lambda_top5),
            proj_w_precond_top5=stack(self._proj_w_precond_top5),
            proj_wb_precond_top5=stack(self._proj_wb_precond_top5),
            proj_w_precond_fixed_top5=stack(self._proj_w_precond_fixed_top5),
            cos_sim_precond_top5=stack(self._cos_sim_precond_top5),
            lambda_precond_top5=stack(self._lambda_precond_top5),
            track_from=np.int64(self.track_from),
            track_until=np.int64(self.track_until),
        )
        tmp_path.replace(out_path)
        print(f"ProjectionTracker: saved {len(self._steps)} steps to {out_path}")
