import torch as T
import torch
import torch.nn as nn
from torch import linalg as LA
import numpy as np
from typing import List, Optional, Sequence

import wandb
from .lobpcg import torch_lobpcg, _maybe_orthonormalize
from torch.func import functional_call

import time
import os
from scipy import stats

from scipy.sparse.linalg import LinearOperator, eigsh

from .gauss_newton import ggn_matvec


__all__ = ['param_vector', 'param_length', 'flatt', 'grads_vector',
           'compute_eigenvalues', 'compute_grad_H_grad',
           'calculate_averaged_grad_H_grad', 'calculate_averaged_grad_H_grad_step',
           'calculate_accuracy',
           'EigenvectorCache', 'HessianVectorProduct', 'create_hessian_vector_product',
           'compute_multiple_eigenvalues_lobpcg',
           'gimme_new_rng', 'gimme_random_subset_idx',
           'compute_grad_gauss_newton_grad',
           'compute_gauss_newton_eigenvalues',
           'compute_gbs_actual_batch', 'compute_gbs_probe_batches',
           'compute_gbs_full_batch']


class EigenvectorCache:
    """
    A cache for storing eigenvectors to enable warm starts in power iteration methods.
    Designed to be compatible with future LOBPCG implementations.
    """
    def __init__(self, max_eigenvectors=5):
        self.max_eigenvectors = max_eigenvectors
        self.eigenvectors = []   # List of eigenvectors for multi-eigenvalue computations
        self.eigenvalues = []    # Corresponding eigenvalues

    def store_eigenvector(self, eigenvector, eigenvalue=None):
        """Store a single eigenvector (and optionally eigenvalue)"""
        if eigenvalue is not None:
            self.eigenvalues = [eigenvalue]
        self.eigenvectors = [eigenvector]

    def store_eigenvectors(self, eigenvectors_list, eigenvalues_list=None):
        """Store multiple eigenvectors (for future LOBPCG compatibility)"""
        self.eigenvectors = [v.detach().clone() for v in eigenvectors_list]
        if eigenvalues_list is not None:
            self.eigenvalues = list(eigenvalues_list)

        # Trim to maximum size
        if len(self.eigenvectors) > self.max_eigenvectors:
            self.eigenvectors = self.eigenvectors[:self.max_eigenvectors]
            if self.eigenvalues:
                self.eigenvalues = self.eigenvalues[:self.max_eigenvectors]

    def get_warm_start_vectors(self, device=None):
        """Get eigenvectors for warm start, optionally moved to specified device"""
        if not self.eigenvectors:
            return None

        if device is not None:
            return [v.to(device) for v in self.eigenvectors]
        return self.eigenvectors

    def clear(self):
        """Clear all cached eigenvectors"""
        self.eigenvectors = []
        self.eigenvalues = []

    def __len__(self):
        return len(self.eigenvectors)

    def __contains__(self, key):
        # For backward compatibility with dict-like access
        return hasattr(self, key) and getattr(self, key) is not None



################################################################################
#                                                                              #
#                               HELPER FUNCTIONS                               #
#                                                                              #
################################################################################


def param_vector(net, clone=True):
    '''
    Returns a vector of all the parameters of the network
    If clone=True, returns a detached clone of the parameters
    '''
    param_vector = T.cat([p.flatten() for p in net.parameters()])
    if clone:
        return param_vector.detach().clone()
    return param_vector

def param_length(net):
    '''
    Returns the number of parameters in the network
    '''
    params = list(net.parameters())
    return sum([p.numel() for p in params])

def flatt(vectors):
    '''
    Flattens a list of vectors into a single vector
    '''
    return T.cat([v.flatten() for v in vectors])


def grads_vector(net):
    # pull out all the gradients from a network as one vector
    grads = []
    for p in net.parameters():
        grads.append(p.grad.flatten().detach().clone())
    return T.cat(grads)


def gimme_new_rng():
    """
    Create a new random number generator with a unique seed.
    """
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    return rng


def gimme_random_subset_idx(dataset_size, subset_size):
    """
    Get random indices for a subset of the dataset.

    Args:
        dataset_size (int): Total size of the dataset.
        subset_size (int): Desired size of the subset.

    Returns:
        Tensor: Random indices for the subset.
    """
    rng = gimme_new_rng()

    shuffle = T.randperm(dataset_size, generator=rng)
    random_idx = shuffle[:subset_size]
    return random_idx


def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy given the model predictions and target labels.

    Args:
        predictions: tensor of shape (num_samples, num_classes) with model predictions
        targets: tensor of shape (num_samples, num_classes) with one-hot encoded labels
                or tensor of shape (num_samples,) with class indices

    Returns:
        accuracy: float representing the accuracy (0.0 to 1.0)
    """
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Get the predicted class (highest value in each row)
        # this is if we have all the classes
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        # Get the predicted class (sign of the prediction)
        # this is if we have only two classes
        pred_classes = torch.sign(predictions).long()



    # Check if targets are one-hot encoded or class indices
    if len(targets.shape) > 1 and targets.shape[1] > 1:
        # One-hot encoded targets
        true_classes = torch.argmax(targets, dim=1)
    else:
        # Class indices (1D tensor)
        if len(targets.shape) == 1:
            true_classes = torch.round(targets).long()
        else:
            true_classes = targets.long()

    # Compare and compute accuracy
    correct = (pred_classes == true_classes).sum().item()
    total = targets.size(0)

    return correct / total


################################################################################
#                                                                              #
#                             HESSIAN-VECTOR PRODUCT                           #
#                                                                              #
################################################################################


class HessianVectorProduct:
    """Callable Hessian-vector product with explicit lifecycle management."""

    def __init__(self,
                 loss,
                 net,
                 params: Optional[Sequence[torch.Tensor]] = None,
                 grads: Optional[Sequence[torch.Tensor]] = None,
                 flat_grads: Optional[torch.Tensor] = None,
                 retain_graph: bool = True):
        if params is None:
            params = list(net.parameters())
        else:
            params = list(params)

        if len(params) == 0:
            raise ValueError("create_hessian_vector_product requires at least one parameter.")

        if grads is None and flat_grads is None:
            grads = torch.autograd.grad(loss, params, create_graph=True)

        if flat_grads is None:
            grads_vector = flatt(grads)
        else:
            grads_vector = flat_grads

        self._params = params
        self._grads = grads
        self._loss_ref = loss  # keep graph alive
        self._retain_graph_default = retain_graph

        grads_vector = grads_vector.view(-1)
        self._grads_vector = grads_vector
        self._device = grads_vector.device
        self._dtype = grads_vector.dtype
        self._numel = grads_vector.numel()
        self._freed = False

    def _ensure_active(self):
        if self._freed:
            raise RuntimeError("HessianVectorProduct.free_memory() has already been called.")

    def _prepare_vec(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.numel() != self._numel:
            raise ValueError("Vector shape mismatch for Hessian-vector product.")
        if vec.device != self._device or vec.dtype != self._dtype:
            vec = vec.to(device=self._device, dtype=self._dtype)
        return vec

    def _apply_single_vector(self, vec: torch.Tensor, retain_flag: bool, create_graph: bool = False) -> torch.Tensor:
        self._ensure_active()
        if vec.ndim != 1:
            raise ValueError(f"Expected 1D vector for Hessian application, got {vec.ndim}D tensor.")
        vec = self._prepare_vec(vec)
        grad_v = torch.dot(self._grads_vector, vec)
        Hv = torch.autograd.grad(grad_v, self._params, retain_graph=retain_flag, create_graph=create_graph)
        return flatt(Hv)

    def __call__(self, v: torch.Tensor, retain_graph_override: Optional[bool] = None, create_graph: bool = False) -> torch.Tensor:
        retain_flag = self._retain_graph_default if retain_graph_override is None else retain_graph_override
        if v.dim() == 1:
            return self._apply_single_vector(v, retain_flag, create_graph=create_graph)
        if v.dim() == 2:
            results = []
            num_vecs = v.shape[1]
            for i in range(num_vecs):
                vi = v[:, i]
                needs_retain = True if retain_flag else (i < num_vecs - 1)
                results.append(self._apply_single_vector(vi, needs_retain, create_graph=create_graph))
            return torch.stack(results, dim=1)
        raise ValueError(f"Input tensor must be 1D or 2D, got {v.dim()}D")

    def free_memory(self):
        if self._freed:
            return
        self._params = None
        self._grads = None
        self._grads_vector = None
        self._loss_ref = None
        self._freed = True


def create_hessian_vector_product(loss,
                                  net,
                                  params: Optional[Sequence[torch.Tensor]] = None,
                                  grads: Optional[Sequence[torch.Tensor]] = None,
                                  flat_grads: Optional[torch.Tensor] = None,
                                  retain_graph: bool = True):
    """
    Create a Hessian-vector product helper for use with LOBPCG and related routines.

    Returns:
        HessianVectorProduct: Callable that also exposes `free_memory()` for manual teardown.
    """
    return HessianVectorProduct(
        loss,
        net,
        params=params,
        grads=grads,
        flat_grads=flat_grads,
        retain_graph=retain_graph,
    )


################################################################################
#                                                                              #
#                             EIGENVALUE FUNCTIONS                             #
#                                                                              #
################################################################################


def compute_eigenvalues(loss,
                        net,
                        k=1,
                        max_iterations=100,
                        reltol=1e-2,
                        init_vectors=None,
                        batched=None,
                        eigenvector_cache=None,
                        return_eigenvectors: bool = False,
                        use_power_iteration: bool = False,
                        use_lanczos: bool = False,
                        ):
    """
    Computes the top-k eigenvalues of the Hessian of the loss function at the current point.

    Uses LOBPCG by default for better performance, with power iteration as fallback for k=1.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    if use_power_iteration and k > 1:
        raise ValueError("Power iteration only supports k=1. Use LOBPCG (default) for k>1.")

    if use_lanczos and use_power_iteration:
        raise ValueError("Cannot use both Lanczos and power iteration simultaneously.")

    device = next(net.parameters()).device

    # Choose method: use LOBPCG by default unless explicitly requested to use power iteration
    if use_power_iteration and k == 1:
        return compute_lambdamax_power_iteration(
            loss, net, max_iterations, reltol, init_vectors,
            eigenvector_cache, return_eigenvectors
        )

    if use_lanczos:
        if use_power_iteration:
            raise ValueError("Lanczos path does not support power iteration mode.")
        n_params = param_length(net)
        if n_params == 0:
            raise ValueError("Model must have parameters to compute eigenvalues.")
        if k >= n_params:
            raise ValueError(f"Lanczos requires k < number of params; got k={k}, n_params={n_params}.")

        hvp = create_hessian_vector_product(loss, net)
        param_dtype = next(net.parameters()).dtype
        np_dtype = np.float64 if param_dtype == torch.float64 else np.float32

        def _apply_operator(vec_torch: torch.Tensor) -> torch.Tensor:
            if vec_torch.ndim != 1 or vec_torch.numel() != n_params:
                raise ValueError("Vector shape mismatch for Lanczos matvec.")
            return hvp(vec_torch).detach()

        def _matvec(vec_np: np.ndarray) -> np.ndarray:
            vec_torch = torch.from_numpy(vec_np).to(device=device, dtype=param_dtype)
            operator_vec = _apply_operator(vec_torch)
            return operator_vec.cpu().numpy().astype(np_dtype, copy=False)

        if n_params == 1:
            basis = torch.ones(n_params, device=device, dtype=param_dtype)
            eigvals_torch = _apply_operator(basis)
            hvp.free_memory()
            if return_eigenvectors:
                if k != 1:
                    raise ValueError("k must be 1 when the parameter space is one-dimensional.")
                return eigvals_torch[0], basis
            return eigvals_torch[0]

        linear_op = LinearOperator(
            shape=(n_params, n_params),
            matvec=_matvec,
            rmatvec=_matvec,
            dtype=np_dtype,
        )
        eigvals_np, eigvecs_np = eigsh(
            linear_op,
            k=k,
            which="LM",
        )
        hvp.free_memory()

        eigvals = torch.from_numpy(eigvals_np.real.astype(np_dtype, copy=False)).to(device=device, dtype=param_dtype)
        eigvecs = torch.from_numpy(eigvecs_np.astype(np_dtype, copy=False)).to(device=device, dtype=param_dtype)

        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        if k == 1:
            if return_eigenvectors:
                return eigvals[0], eigvecs[:, 0]
            return eigvals[0]
        if return_eigenvectors:
            return eigvals, eigvecs
        return eigvals

    else:
        # Use LOBPCG method (default)
        eigenvalues, eigenvectors = compute_multiple_eigenvalues_lobpcg(
            loss, net, k, max_iterations, reltol, init_vectors,
            eigenvector_cache, return_eigenvectors=True
        )

        if k == 1:
            eigenvalue = eigenvalues[0]
            if return_eigenvectors:
                return eigenvalue, eigenvectors[:, 0]
            else:
                return eigenvalue
        else:
            if return_eigenvectors:
                return eigenvalues, eigenvectors
            else:
                return eigenvalues


def _run_lobpcg_with_operator(
    operator,
    net,
    k: int,
    max_iterations: int,
    reltol: float,
    init_vectors: Optional[torch.Tensor],
    eigenvector_cache: Optional[EigenvectorCache],
    return_eigenvectors: bool,
):
    param_example = next(net.parameters(), None)
    if param_example is None:
        raise ValueError("Model must have parameters to compute eigenvalues.")
    device = param_example.device
    dtype = param_example.dtype
    n_params = param_length(net)

    if init_vectors is not None:
        X = init_vectors
        if X.shape[1] != k or X.shape[0] != n_params:
            raise ValueError(f"init_vectors must have shape [{n_params}, {k}], got {X.shape}")
    elif eigenvector_cache is not None and len(eigenvector_cache) > 0:
        cached_vectors = eigenvector_cache.get_warm_start_vectors(device)
        if cached_vectors:
            n_cached = min(len(cached_vectors), k)
            chosen = [cached_vectors[i] for i in range(n_cached)]
            if n_cached < k:
                padding = torch.randn(n_params, k - n_cached, device=device, dtype=dtype)
                chosen.extend([padding[:, j] for j in range(padding.shape[1])])
            X = torch.stack(chosen, dim=1)
        else:
            X = torch.randn(n_params, k, device=device, dtype=dtype)
    else:
        X = torch.randn(n_params, k, device=device, dtype=dtype)

    X = X.to(device=device, dtype=dtype).reshape(n_params, k)
    tol = reltol / (20 * max(n_params, 1))

    eigenvalues = None
    eigenvectors = None
    iterations = None
    try:
        eigenvalues, eigenvectors, iterations = torch_lobpcg(
            operator, X, max_iter=max_iterations, tol=tol
        )
    finally:
        pass

    try:
        wandb.log({"lobpcg_iterations": iterations}, commit=False)
    except Exception:
        pass

    if eigenvector_cache is not None:
        eigenvector_list = [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]
        eigenvector_cache.store_eigenvectors(eigenvector_list, eigenvalues.tolist())

    if return_eigenvectors:
        return eigenvalues, eigenvectors
    return eigenvalues


def compute_multiple_eigenvalues_lobpcg(loss, net, k=5, max_iterations=100, reltol=1e-2,
                                       init_vectors=None, eigenvector_cache=None,
                                       return_eigenvectors=False):
    """
    Compute multiple eigenvalues of the Hessian using LOBPCG algorithm.
    """
    hessian_matvec = create_hessian_vector_product(loss, net)
    try:
        return _run_lobpcg_with_operator(
            hessian_matvec,
            net,
            k,
            max_iterations,
            reltol,
            init_vectors,
            eigenvector_cache,
            return_eigenvectors,
        )
    finally:
        hessian_matvec.free_memory()



def compute_lambdamax_power_iteration(loss, net, max_iterations, reltol, init_vector,
                                       eigenvector_cache, return_eigenvector):
    """Power iteration implementation of the maximum eigenvalue of the Hessian."""
    device = next(net.parameters()).device

    # compute gradient and keep it for repeated Hessian-vector products
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    hessian_vector_product = create_hessian_vector_product(
        loss,
        net,
        params=params,
        grads=grads,
        retain_graph=True,
    )

    try:
        size = param_length(net)

        # Initialize vector with priority: init_vector > cached eigenvector > gradient
        if init_vector is not None:
            v = init_vector
        elif eigenvector_cache is not None:
            if isinstance(eigenvector_cache, EigenvectorCache):
                if len(eigenvector_cache) > 0:
                    cached_v = eigenvector_cache.eigenvectors[0]
                    if cached_v.device != device:
                        cached_v = cached_v.to(device)
                    v = cached_v.detach()
                else:
                    v = T.randn(size, device=device)
            elif isinstance(eigenvector_cache, dict) and 'eigenvector' in eigenvector_cache:
                cached_v = eigenvector_cache['eigenvector']
                if cached_v.device != device:
                    cached_v = cached_v.to(device)
                v = cached_v.detach()
            else:
                v = T.randn(size, device=device)
        else:
            v = T.randn(size, device=device)

        with torch.no_grad():
            v = v / T.linalg.norm(v)

        v = v.detach()
        eigenval = 0.0
        for i in range(max_iterations):
            Hv = hessian_vector_product(v).detach()

            v = v.detach()
            with T.no_grad():
                rayleigh_quotient = T.dot(Hv, v) / T.dot(v, v)
                eigenval = rayleigh_quotient
                if T.abs(rayleigh_quotient) < 1e-12:
                    break

                residual = Hv - rayleigh_quotient * v
                resid_norm = T.linalg.norm(residual)
                if resid_norm / T.abs(rayleigh_quotient) < reltol:
                    break

                v = Hv / T.linalg.norm(Hv)

        # Log the number of iterations to wandb
        try:
            wandb.log({"power_iteration_iterations": i + 1}, commit=False)
        except:
            pass

        # Store the final eigenvector in cache for future warm starts
        if eigenvector_cache is not None:
            if isinstance(eigenvector_cache, EigenvectorCache):
                eigenvector_cache.store_eigenvector(v, eigenval)
            else:
                raise ValueError("eigenvector_cache must be an instance of EigenvectorCache")

        results = [eigenval]
        if return_eigenvector:
            results.append(v.detach())
    finally:
        hessian_vector_product.free_memory()

    if len(results) == 1:
        return results[0]
    return tuple(results)


################################################################################
#                                                                              #
#                         GRAD-H-GRAD (BATCH SHARPNESS)                        #
#                                                                              #
################################################################################


def compute_grad_H_grad(loss, net, grad_already_there: bool = False,
                        return_ghg_gg_separately: bool = False):
    """
    Computes g^T H g / ||g||², the Rayleigh quotient of the Hessian H and gradient g.
    """

    device = next(net.parameters()).device

    params = list(net.parameters())
    if not grad_already_there:
        grads = torch.autograd.grad(loss, params, create_graph=True)
    else:
        grads = [p.grad for p in params]
        if any(g is None or not g.requires_grad for g in grads):
            grads = torch.autograd.grad(loss, params, create_graph=True)

    grads_vector = flatt(grads)
    step_vector = grads_vector.detach()

    hvp = create_hessian_vector_product(
        loss,
        net,
        params=params,
        grads=grads,
        flat_grads=grads_vector,
    )
    try:
        Hv = hvp(step_vector, retain_graph_override=False).detach()
    finally:
        hvp.free_memory()

    if return_ghg_gg_separately:
        return T.dot(step_vector, Hv), T.dot(step_vector, step_vector)
    return T.dot(step_vector, Hv) / T.dot(step_vector, step_vector)


def calculate_averaged_grad_H_grad(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              eps = 0.005,
                              expectation_inside = False,
                              with_replacement = False,
                              return_confidence_interval: bool = False,
                              confidence_level: float = 0.95,
                              use_gauss_newton: bool = False,
                              gauss_newton_loss_type: Optional[str] = None,
                              ):
    """
    Computes E[g_b H_b g_b / ||g_b||²], which represents batch sharpness.
    """
    if use_gauss_newton:
        gn_loss_type = _infer_gauss_newton_loss_type(loss_fn, gauss_newton_loss_type)
    else:
        gn_loss_type = None

    gHg_vals = []
    norm_g_vals = []

    x_vals = gHg_vals
    y_vals = norm_g_vals

    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)

    for i in range(n_estimates):
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        if with_replacement:
            random_idx = T.randint(0, len(X), (batch_size,), generator=rng)

        if batch_size > 128:
            torch.cuda.empty_cache()

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]

        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        if use_gauss_newton:
            gHg, norm_g = compute_grad_gauss_newton_grad(
                loss,
                net,
                X_batch,
                Y_batch,
                loss_type=gn_loss_type,
                return_gcg_gg_separately=True,
            )
        else:
            gHg, norm_g = compute_grad_H_grad(loss, net, return_ghg_gg_separately=True)
        gHg = gHg.item()
        norm_g = norm_g.item()


        gHg_vals.append(gHg)
        norm_g_vals.append(norm_g)

        if i < min_estimates:
            continue

        mean_x, mean_y = np.mean(x_vals), np.mean(y_vals)
        var_x,  var_y  = np.var(x_vals, ddof=1), np.var(y_vals, ddof=1)
        cov_xy = np.cov(x_vals, y_vals, ddof=1)[0, 1]

        R = mean_x / mean_y

        var_R = (var_x / mean_y**2
                 - 2 * cov_xy * mean_x / mean_y**3
                 + var_y * mean_x**2 / mean_y**4) / i

        rse = np.sqrt(var_R) / abs(R)  # relative standard error

        if rse < eps:                    # stopping rule
            break


    num_samples = len(gHg_vals)

    try:
        wandb.log({"number_of_gHg_estimates": num_samples}, commit=False)
    except:
        pass


    if num_samples == 0:
        raise RuntimeError("calculate_averaged_grad_H_grad received no samples; check dataset and parameters.")

    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    alpha = 1 - confidence_level

    if expectation_inside:
        mean_x = float(np.mean(gHg_vals))
        mean_y = float(np.mean(norm_g_vals))
        if mean_y == 0.0:
            raise ZeroDivisionError("Mean squared gradient is zero; cannot compute batch sharpness.")

        result = mean_x / mean_y

        if not return_confidence_interval:
            return result

        if num_samples < 2:
            stderr = 0.0
            ci = (result, result)
        else:
            var_x = float(np.var(gHg_vals, ddof=1))
            var_y = float(np.var(norm_g_vals, ddof=1))
            cov_xy = float(np.cov(gHg_vals, norm_g_vals, ddof=1)[0, 1])
            var_R = (
                var_x / (mean_y ** 2)
                - 2 * cov_xy * mean_x / (mean_y ** 3)
                + var_y * (mean_x ** 2) / (mean_y ** 4)
            ) / num_samples
            var_R = max(var_R, 0.0)
            stderr = float(np.sqrt(var_R))
            t_multiplier = stats.t.ppf(1 - alpha / 2, df=num_samples - 1) if num_samples > 1 else 0.0
            if not np.isfinite(t_multiplier):
                t_multiplier = 0.0
            half_width = float(t_multiplier * stderr)
            ci = (result - half_width, result + half_width)

        return {
            "mean": result,
            "ci": ci,
            "stderr": stderr,
            "confidence_level": confidence_level,
            "num_samples": num_samples,
        }

    gHg_normalized = np.array(gHg_vals) / np.array(norm_g_vals)
    result = float(np.mean(gHg_normalized))

    if not return_confidence_interval:
        return result

    if num_samples < 2:
        stderr = 0.0
        ci = (result, result)
    else:
        std = float(np.std(gHg_normalized, ddof=1))
        stderr = float(std / np.sqrt(num_samples))
        t_multiplier = stats.t.ppf(1 - alpha / 2, df=num_samples - 1)
        if not np.isfinite(t_multiplier):
            t_multiplier = 0.0
        half_width = float(t_multiplier * stderr)
        ci = (result - half_width, result + half_width)

    return {
        "mean": result,
        "ci": ci,
        "stderr": stderr,
        "confidence_level": confidence_level,
        "num_samples": num_samples,
    }


def calculate_averaged_grad_H_grad_step(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 1000,
                              min_estimates = 10,
                              eps = 0.005,
                              log_the_expectation_outside = False,
                              return_ghg_gg_separately = False,
                              with_replacement = False,
                              return_confidence_interval: bool = False,
                              confidence_level: float = 0.95,
                              use_gauss_newton: bool = False,
                              gauss_newton_loss_type: Optional[str] = None,
                              ):
    """Backward-compatible wrapper for the batch sharpness estimator E[gHg/g²]."""
    if return_ghg_gg_separately:
        raise NotImplementedError("Returning gHg and g² separately is not supported in this refactor.")

    result = calculate_averaged_grad_H_grad(
        net=net,
        X=X,
        Y=Y,
        loss_fn=loss_fn,
        batch_size=batch_size,
        n_estimates=n_estimates,
        min_estimates=min_estimates,
        eps=eps,
        expectation_inside=False,
        with_replacement=with_replacement,
        return_confidence_interval=return_confidence_interval,
        confidence_level=confidence_level,
        use_gauss_newton=use_gauss_newton,
        gauss_newton_loss_type=gauss_newton_loss_type,
    )

    return result


################################################################################
#                                                                              #
#                    GENERALIZED BATCH SHARPNESS (GBS)                         #
#                                                                              #
################################################################################


def _averaged_hvp_power_iteration(net, X, Y, loss_fn, batch_indices_list, n_params, device, num_iters):
    """Find top eigenvector of the averaged Hessian via power iteration.

    For each power iteration step, loops over all batches, computes H_b @ v,
    averages, and normalizes. Returns a unit vector u_avg.
    """
    v = torch.randn(n_params, device=device)
    v = v / v.norm()
    n_batches = len(batch_indices_list)
    for _ in range(num_iters):
        Hv_sum = torch.zeros(n_params, device=device)
        for idx in batch_indices_list:
            X_b = X[idx]
            Y_b = Y[idx]
            loss_b = loss_fn(net(X_b).squeeze(dim=-1), Y_b)
            params_b = list(net.parameters())
            grads_b = torch.autograd.grad(loss_b, params_b, create_graph=True)
            grads_b_flat = flatt(grads_b)
            hvp_b = create_hessian_vector_product(
                loss_b, net, params=params_b, grads=grads_b, flat_grads=grads_b_flat,
            )
            try:
                Hv_b = hvp_b(v, retain_graph_override=False).detach()
            finally:
                hvp_b.free_memory()
            Hv_sum += Hv_b
        Hv_avg = Hv_sum / n_batches
        norm = Hv_avg.norm()
        if norm < 1e-12:
            break
        v = Hv_avg / norm
    return v


def _power_iteration_top_eigenvector(hvp, n_params, device, num_iters):
    """Find top eigenvector of the Hessian via power iteration using an HVP operator."""
    v = torch.randn(n_params, device=device)
    v = v / v.norm()
    for _ in range(num_iters):
        Hv = hvp(v, retain_graph_override=True).detach()
        norm = Hv.norm()
        if norm < 1e-12:
            break
        v = Hv / norm
    return v


def compute_gbs_actual_batch(net, X_batch, Y_batch, loss_fn, optimizer_wrapper, power_iters=50, compute_u=True):
    """Compute GBS quantities on the actual training batch.

    u is the top eigenvector of this single batch's Hessian.
    Returns (A, A_u, B, B_u).
    """
    loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads_flat = flatt(grads)

    s_b = optimizer_wrapper.compute_step_direction(grads_flat, params)

    hvp = create_hessian_vector_product(
        loss, net, params=params, grads=grads, flat_grads=grads_flat,
    )
    try:
        Hs = hvp(s_b, retain_graph_override=True)
        A = T.dot(grads_flat, s_b).item()
        B = T.dot(s_b, Hs).item()
    finally:
        hvp.free_memory()

    A_u, B_u = float('nan'), float('nan')
    if compute_u:
        # Fresh forward pass for power iteration
        loss2 = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)
        params2 = list(net.parameters())
        grads2 = torch.autograd.grad(loss2, params2, create_graph=True)
        grads2_flat = flatt(grads2)
        hvp2 = create_hessian_vector_product(
            loss2, net, params=params2, grads=grads2, flat_grads=grads2_flat,
        )
        try:
            u = _power_iteration_top_eigenvector(hvp2, s_b.numel(), s_b.device, power_iters)
            s_u = T.dot(s_b, u).item() * u
            Hs_u = hvp2(s_u)
            A_u = T.dot(grads_flat.detach(), s_u).item()
            B_u = T.dot(s_u, Hs_u).item()
        finally:
            hvp2.free_memory()

    out_actual   = B    / (-A)    if abs(A)    > 1e-12 else float('nan')
    out_actual_u = B_u  / (-A_u)  if abs(A_u)  > 1e-12 else float('nan')
    return A, A_u, B, B_u, out_actual, out_actual_u


def compute_gbs_probe_batches(net, X, Y, loss_fn, optimizer_wrapper, batch_size, n_probe=128, power_iters=50, compute_u=True):
    """Compute averaged GBS quantities over probe batches.

    Two-pass approach:
      Pass 1: compute u_avg = top eigenvector of the averaged Hessian over all probe batches.
              (skipped when compute_u=False)
      Pass 2: for each probe batch compute A, B (using per-batch s, g) and
              A_u, B_u (using u_avg as the shared eigenvector direction).

    Returns (mean_A, mean_A_u, mean_B, mean_B_u).
    """
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)

    all_batch_indices = [T.randperm(len(X), generator=rng)[:batch_size] for _ in range(n_probe)]

    # Pass 1: u_avg via averaged HVP power iteration
    if compute_u:
        n_params = sum(p.numel() for p in net.parameters())
        device = next(net.parameters()).device
        u_avg = _averaged_hvp_power_iteration(
            net, X, Y, loss_fn, all_batch_indices, n_params, device, power_iters,
        )
    else:
        u_avg = None

    # Pass 2: compute quantities for each probe batch
    A_vals, A_u_vals, B_vals, B_u_vals = [], [], [], []

    for random_idx in all_batch_indices:
        if batch_size > 128:
            torch.cuda.empty_cache()

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]

        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)
        params = list(net.parameters())
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grads_flat = flatt(grads)
        s_b = optimizer_wrapper.compute_step_direction(grads_flat, params)

        hvp = create_hessian_vector_product(
            loss, net, params=params, grads=grads, flat_grads=grads_flat,
        )
        try:
            Hs = hvp(s_b, retain_graph_override=True)
            A_val = T.dot(grads_flat, s_b).item()
            B_val = T.dot(s_b, Hs).item()
        finally:
            hvp.free_memory()

        A_u_val, B_u_val = float('nan'), float('nan')
        if compute_u:
            # A_u, B_u use u_avg — fresh forward pass
            loss2 = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)
            params2 = list(net.parameters())
            grads2 = torch.autograd.grad(loss2, params2, create_graph=True)
            grads2_flat = flatt(grads2)
            hvp2 = create_hessian_vector_product(
                loss2, net, params=params2, grads=grads2, flat_grads=grads2_flat,
            )
            try:
                s_u = T.dot(s_b, u_avg).item() * u_avg
                Hs_u = hvp2(s_u)
                A_u_val = T.dot(grads_flat.detach(), s_u).item()
                B_u_val = T.dot(s_u, Hs_u).item()
            finally:
                hvp2.free_memory()

        A_vals.append(A_val)
        A_u_vals.append(A_u_val)
        B_vals.append(B_val)
        B_u_vals.append(B_u_val)

    out_probe   = float(np.mean(np.array(B_vals)   / np.array([-a for a in A_vals])))
    out_probe_u = float(np.mean(np.array(B_u_vals) / np.array([-a for a in A_u_vals]))) if compute_u else float('nan')
    return (float(np.mean(A_vals)), float(np.mean(A_u_vals)) if compute_u else float('nan'),
            float(np.mean(B_vals)), float(np.mean(B_u_vals)) if compute_u else float('nan'),
            out_probe, out_probe_u)


def compute_gbs_full_batch(net, X_full, Y_full, X_batch, Y_batch, loss_fn, optimizer_wrapper, u, compute_u=True):
    """Compute GBS quantities using actual batch for g/s, full-batch Hessian for B.

    u is the pre-computed top eigenvector of the full-batch Hessian (e.g. from lambda_max).
    Returns (A_full, A_u_full, B_full, B_u_full, out_full, out_full_u).
    """
    # Get g and s from actual training batch
    loss_batch = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)
    params = list(net.parameters())
    grads_batch = T.autograd.grad(loss_batch, params, create_graph=False)
    grads_batch_flat = flatt(grads_batch).detach()
    s = optimizer_wrapper.compute_step_direction(grads_batch_flat, params)

    # Full-batch HVP for B computation
    loss_full = loss_fn(net(X_full).squeeze(dim=-1), Y_full)
    params_full = list(net.parameters())
    grads_full = T.autograd.grad(loss_full, params_full, create_graph=True)
    grads_full_flat = flatt(grads_full)
    hvp = create_hessian_vector_product(
        loss_full, net, params=params_full, grads=grads_full, flat_grads=grads_full_flat,
    )
    A_u_full, B_u_full = float('nan'), float('nan')
    try:
        retain = compute_u and (u is not None)
        Hs = hvp(s, retain_graph_override=retain)
        A_full = T.dot(grads_batch_flat, s).item()
        B_full = T.dot(s, Hs).item()

        if compute_u and u is not None:
            s_u = T.dot(s, u).item() * u
            Hs_u = hvp(s_u)
            A_u_full = T.dot(grads_batch_flat, s_u).item()
            B_u_full = T.dot(s_u, Hs_u).item()
    finally:
        hvp.free_memory()

    out_full   = B_full   / (-A_full)   if abs(A_full)   > 1e-12 else float('nan')
    out_full_u = B_u_full / (-A_u_full) if abs(A_u_full) > 1e-12 else float('nan')
    return A_full, A_u_full, B_full, B_u_full, out_full, out_full_u


################################################################################
#                                                                              #
#                              GAUSS-NEWTON OPS                                #
#                                                                              #
################################################################################


_GAUSS_NEWTON_LOSS_ALIASES = {
    'ce': 'ce',
    'cross_entropy': 'ce',
    'crossentropyloss': 'ce',
    'categorical_crossentropy': 'ce',
    'mse': 'mse',
    'mean_squared_error': 'mse',
    'squared': 'mse',
}


def _normalize_gauss_newton_loss_type(loss_type: str) -> str:
    if not isinstance(loss_type, str):
        raise TypeError("Gauss-Newton loss type override must be a string.")
    key = loss_type.replace('-', '_').lower()
    normalized = _GAUSS_NEWTON_LOSS_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"Unsupported Gauss-Newton loss type '{loss_type}'. "
            "Supported values: 'ce', 'cross_entropy', 'mse'."
        )
    return normalized


def _infer_gauss_newton_loss_type(loss_fn, override: Optional[str] = None) -> str:
    if override is not None:
        return _normalize_gauss_newton_loss_type(override)

    if isinstance(loss_fn, nn.CrossEntropyLoss):
        return 'ce'
    if loss_fn.__class__.__name__ == 'SquaredLoss' or isinstance(loss_fn, nn.MSELoss):
        return 'mse'

    raise ValueError(
        "Unable to infer Gauss-Newton loss type from the provided loss_fn. "
        "Pass gauss_newton_loss_type explicitly ('ce' or 'mse')."
    )


def compute_grad_gauss_newton_grad(
    loss,
    net,
    X_batch,
    Y_batch,
    *,
    loss_type: str,
    grad_already_there: bool = False,
    return_gcg_gg_separately: bool = False,
):
    """
    Computes g^T G g / ||g||² where G is the Gauss-Newton matrix.
    """
    normalized_loss_type = _normalize_gauss_newton_loss_type(loss_type)

    params = list(net.parameters())
    if not params:
        raise ValueError("compute_grad_gauss_newton_grad requires a model with parameters.")

    if not grad_already_there:
        grads = torch.autograd.grad(loss, params, create_graph=False)
    else:
        grads = [p.grad for p in params]
        if any(g is None for g in grads):
            grads = torch.autograd.grad(loss, params, create_graph=False)

    grads_vector = flatt(grads).detach()
    if grads_vector.numel() == 0:
        raise ValueError("Gradient vector is empty; ensure the model has trainable parameters.")

    Gg = ggn_matvec(
        model=net,
        x=X_batch,
        y=Y_batch,
        v_flat=grads_vector,
        loss=normalized_loss_type,
        average_over_batch=True,
    ).detach()

    numerator = torch.dot(grads_vector, Gg)
    denominator = torch.dot(grads_vector, grads_vector)
    if return_gcg_gg_separately:
        return numerator, denominator
    return numerator / denominator


class GaussNewtonVectorProduct:
    """
    Callable Gauss-Newton vector product that mimics the HessianVectorProduct interface.
    """

    def __init__(
        self,
        net: nn.Module,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        loss_type: str,
        *,
        average_over_batch: bool = True,
    ):
        param_example = next(net.parameters(), None)
        if param_example is None:
            raise ValueError("Gauss-Newton vector product requires a model with parameters.")

        self._net = net
        self._loss_type = _normalize_gauss_newton_loss_type(loss_type)
        self._average_over_batch = average_over_batch

        device = param_example.device
        dtype = param_example.dtype
        self._X_batch = X_batch.to(device=device, dtype=dtype)
        self._Y_batch = Y_batch.to(device=device)

        self._numel = param_length(net)
        self._device = device
        self._dtype = dtype
        self._freed = False

    def _ensure_active(self):
        if self._freed:
            raise RuntimeError("GaussNewtonVectorProduct.free_memory() has already been called.")

    def _prepare_vec(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.numel() != self._numel:
            raise ValueError(
                f"Vector shape mismatch for Gauss-Newton product: expected {self._numel}, got {vec.numel()}."
            )
        if vec.device != self._device or vec.dtype != self._dtype:
            vec = vec.to(device=self._device, dtype=self._dtype)
        return vec

    def _apply_single_vector(self, vec: torch.Tensor) -> torch.Tensor:
        self._ensure_active()
        vec = self._prepare_vec(vec)
        return ggn_matvec(
            model=self._net,
            x=self._X_batch,
            y=self._Y_batch,
            v_flat=vec,
            loss=self._loss_type,
            average_over_batch=self._average_over_batch,
        ).detach()

    def __call__(self, v: torch.Tensor, retain_graph_override: Optional[bool] = None) -> torch.Tensor:
        if v.dim() == 1:
            return self._apply_single_vector(v)
        if v.dim() == 2:
            cols = []
            for i in range(v.shape[1]):
                cols.append(self._apply_single_vector(v[:, i]))
            return torch.stack(cols, dim=1)
        raise ValueError(f"Gauss-Newton operator expects 1D or 2D tensor, got {v.dim()}D.")

    def free_memory(self):
        if self._freed:
            return
        self._X_batch = None
        self._Y_batch = None
        self._net = None
        self._freed = True


def _compute_gauss_newton_power_iteration(
    operator: GaussNewtonVectorProduct,
    net: nn.Module,
    max_iterations: int,
    reltol: float,
    init_vector: Optional[torch.Tensor],
    eigenvector_cache: Optional[EigenvectorCache],
    return_eigenvector: bool,
):
    device = next(net.parameters()).device
    dtype = next(net.parameters()).dtype
    size = param_length(net)

    if init_vector is not None:
        v = init_vector.to(device=device, dtype=dtype)
    elif eigenvector_cache is not None and len(eigenvector_cache) > 0:
        cached = eigenvector_cache.get_warm_start_vectors(device)
        if cached:
            v = cached[0].detach()
        else:
            v = torch.randn(size, device=device, dtype=dtype)
    else:
        v = torch.randn(size, device=device, dtype=dtype)

    with torch.no_grad():
        if torch.linalg.norm(v) == 0:
            raise ValueError("Initialization vector must be non-zero.")
        v = v / torch.linalg.norm(v)

    v = v.detach()
    eigenval = torch.tensor(0.0, device=device, dtype=dtype)
    for i in range(max_iterations):
        Gv = operator(v).detach()
        v = v.detach()
        with torch.no_grad():
            denom = torch.dot(v, v)
            if denom.abs() < 1e-20:
                break
            rayleigh = torch.dot(v, Gv) / denom
            eigenval = rayleigh

            residual = Gv - rayleigh * v
            resid_norm = torch.linalg.norm(residual)
            if torch.abs(rayleigh) < 1e-12 or (resid_norm / torch.abs(rayleigh) < reltol):
                break

            Gv_norm = torch.linalg.norm(Gv)
            if Gv_norm.item() == 0:
                break
            v = Gv / Gv_norm

    iterations_run = (i + 1) if "i" in locals() else 0
    try:
        wandb.log({"power_iteration_iterations": iterations_run}, commit=False)
    except Exception:
        pass

    if eigenvector_cache is not None:
        eigenvector_cache.store_eigenvector(v.detach(), eigenval)

    if return_eigenvector:
        return eigenval, v.detach()
    return eigenval


def compute_gauss_newton_eigenvalues(
    net: nn.Module,
    X_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    loss_fn,
    *,
    k: int = 1,
    max_iterations: int = 100,
    reltol: float = 1e-2,
    init_vectors: Optional[torch.Tensor] = None,
    eigenvector_cache: Optional[EigenvectorCache] = None,
    return_eigenvectors: bool = False,
    use_power_iteration: bool = False,
    use_lanczos: bool = False,
    gauss_newton_loss_type: Optional[str] = None,
):
    """
    Compute the top-k eigenvalues of the Gauss-Newton matrix on a specific mini-batch.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    if X_batch.numel() == 0:
        raise ValueError("X_batch must contain at least one sample.")
    if use_power_iteration and k > 1:
        raise ValueError("Power iteration only supports k=1 for Gauss-Newton eigenvalues.")
    if use_lanczos and use_power_iteration:
        raise ValueError("Cannot combine Lanczos with power iteration.")

    loss_type = _infer_gauss_newton_loss_type(loss_fn, gauss_newton_loss_type)
    gn_operator = GaussNewtonVectorProduct(
        net,
        X_batch,
        Y_batch,
        loss_type=loss_type,
        average_over_batch=True,
    )

    try:
        if use_power_iteration and k == 1:
            init_vector_1d = None
            if init_vectors is not None:
                if isinstance(init_vectors, torch.Tensor) and init_vectors.ndim == 1:
                    init_vector_1d = init_vectors
                else:
                    raise ValueError("init_vectors must be a 1D tensor when using power iteration.")
            return _compute_gauss_newton_power_iteration(
                gn_operator,
                net,
                max_iterations,
                reltol,
                init_vector_1d,
                eigenvector_cache,
                return_eigenvectors,
            )

        # Use LOBPCG
        return _run_lobpcg_with_operator(
            gn_operator,
            net,
            k,
            max_iterations,
            reltol,
            init_vectors,
            eigenvector_cache,
            return_eigenvectors,
        )
    finally:
        gn_operator.free_memory()
