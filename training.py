import torch as T
import torch
import torch.nn as nn
import os
import sys
import numpy as np
from pathlib import Path
import math
import time
import json
from tqdm import tqdm

from utils.data import prepare_dataset, get_dataset_presets
from utils.nets import SquaredLoss, MLP, CNN, prepare_net, initialize_net, get_model_presets
from utils.nets import ResNet, WideResNet, WideResNetNoBN
from utils.storage import initialize_folders, write_json_atomic
from utils.measure import (
    EigenvectorCache,
    param_length,
    gimme_random_subset_idx,
    compute_eigenvalues,
    compute_grad_H_grad,
    calculate_averaged_grad_H_grad_step,
    compute_gbs_probe_batches,
    calculate_accuracy,
)
from utils.frequency import frequency_calculator, MeasurementContext
from utils.projection_tracker import ProjectionTracker


# -------------------------------------
# Section: Measurement Runner
# -------------------------------------

class MeasurementRunner:
    """Centralized measurement orchestration for the training loop (simplified)."""

    def __init__(
        self,
        *,
        net,
        optimizer,
        loss_fn,
        full_inputs,
        measurements,
        device,
        batch_size,
        save_dir,
        eigenvector_cache,
        num_eigenvalues,
        use_power_iteration,
        probe_samples,
        power_iters,
        compute_distributions,
        distributions_file,
        compute_quantities_with_uB=True,
        measurement_batch_size_cap=None,
    ):
        self.net = net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.X, self.Y = full_inputs
        self.measurements = measurements
        self.device = device
        self.batch_size = batch_size
        self.eigenvector_cache = eigenvector_cache
        self.num_eigenvalues = num_eigenvalues
        self.use_power_iteration = use_power_iteration
        self.probe_samples = probe_samples
        self.power_iters = power_iters
        self.compute_distributions = compute_distributions
        self.distributions_file = distributions_file
        self.compute_quantities_with_uB = compute_quantities_with_uB
        self.measurement_batch_size_cap = measurement_batch_size_cap
        self._dist_records = []
        # Cached subset for lmax power iteration. Drawn once on first firing and
        # reused so warm-started power iteration stays effective across firings.
        self._lmax_subset_X: torch.Tensor | None = None
        self._lmax_subset_Y: torch.Tensor | None = None

        self.eigenvalues_log = []
        if 'lmax' in measurements and num_eigenvalues > 1:
            eigenvalues_path = save_dir / 'eigenvalues.json'
            self.eigenvalues_file = open(eigenvalues_path, 'w')
            self.eigenvalues_file.write('[\n')
        else:
            self.eigenvalues_file = None

    def _has_small_gpu(self):
        return (
            str(self.device).startswith('cuda')
            and torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3
        )

    def _is_vit(self):
        return self.net.__class__.__name__ == 'ViT'

    def _measurement_batch_size(self):
        if self.measurement_batch_size_cap is not None:
            return min(self.batch_size, int(self.measurement_batch_size_cap))
        if self._is_vit() and self._has_small_gpu():
            return min(self.batch_size, 128)
        return self.batch_size

    def _full_subset_cap(self):
        lmax_max_size = 4096
        if self._has_small_gpu():
            if isinstance(self.net, CNN):
                lmax_max_size = 2560
            if isinstance(self.net, ResNet):
                lmax_max_size = 512
            if isinstance(self.net, WideResNet) or isinstance(self.net, WideResNetNoBN):
                lmax_max_size = 1024
            if self._is_vit():
                lmax_max_size = 1024
        return lmax_max_size

    def close(self):
        if self._dist_records:
            keys = [k for k in self._dist_records[0] if k != 'step']
            payload = {'steps': np.array([r['step'] for r in self._dist_records])}
            for k in keys:
                arrs = [np.asarray(r[k], dtype=np.float64) for r in self._dist_records]
                # Scalar entries (e.g. gnorm_full) stack to a 1-D per-step vector.
                if all(a.ndim == 0 for a in arrs):
                    payload[k] = np.stack(arrs)
                    continue
                # Per-step arrays may be ragged because the probe loop has an
                # RSE early-exit. Pad with NaN to the max length.
                arrs = [np.atleast_1d(a) for a in arrs]
                max_len = max(a.shape[0] for a in arrs)
                padded = np.full((len(arrs), max_len), np.nan, dtype=np.float64)
                for i, a in enumerate(arrs):
                    padded[i, :a.shape[0]] = a
                payload[k] = padded
            np.savez(self.distributions_file, **payload)
        if self.eigenvalues_file is not None:
            self.eigenvalues_file.write('\n]')
            self.eigenvalues_file.close()

    def collect(
        self,
        *,
        ctx,
        optimizer,
        X_batch,
        Y_batch,
        epoch,
        step_in_epoch,
        step_number,
    ):
        metrics = {
            'step_sharpness': np.nan,
            'batch_sharpness': np.nan,
            'A': np.nan,
            'B': np.nan,
            'GBS': np.nan,
            'SBS': np.nan,
            'A_u': np.nan,
            'B_u': np.nan,
            'GBS_u': np.nan,
            'A_g': np.nan,
            'B_g': np.nan,
            'GBS_g': np.nan,
            'A_gfull': np.nan,
            'B_gfull': np.nan,
            'GBS_gfull': np.nan,
            'A_ufull': np.nan,
            'B_ufull': np.nan,
            'GBS_ufull': np.nan,
            'A_cos_sBgB': np.nan,
            'GBS_cos_sBgB': np.nan,
            'A_cos_sBgfull': np.nan,
            'GBS_cos_sBgfull': np.nan,
            'A_cos_gBgfull': np.nan,
            'GBS_cos_gBgfull': np.nan,
            'full_accuracy': np.nan,
            'full_loss': np.nan,
            'lmax': np.nan,
            'all_eigenvalues': None,
        }

        epoch_loss_update = None

        # ----- Batch sharpness (expected Rayleigh quotient) -----
        if 'batch_sharpness' in self.measurements:
            if frequency_calculator.should_measure('batch_sharpness', ctx):
                metrics['batch_sharpness'] = calculate_averaged_grad_H_grad_step(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    batch_size=self._measurement_batch_size(),
                    n_estimates=self.probe_samples,
                    min_estimates=20,
                    eps=0.005,
                )

        # ----- Instantaneous step sharpness (current-batch Rayleigh quotient) -----
        if 'step_sharpness' in self.measurements:
            if frequency_calculator.should_measure('step_sharpness', ctx):
                self.net.zero_grad()
                preds = self.net(X_batch).squeeze(dim=-1)
                loss = self.loss_fn(preds, Y_batch)
                metrics['step_sharpness'] = compute_grad_H_grad(loss, self.net).item()

        # ----- Eigenvalues/Lambda max (full batch) — runs before probe GBS so u_full is fresh -----
        lmax_now = False
        if 'lmax' in self.measurements:
            measurement_type = 'full_batch_lambda_max'
            lmax_now = frequency_calculator.should_measure(measurement_type, ctx)

        if lmax_now:
            if str(self.device).startswith('cuda'):
                torch.cuda.empty_cache()
            optimizer.zero_grad()

            lmax_max_size = self._full_subset_cap()

            if self._lmax_subset_X is None:
                if len(self.X) > lmax_max_size:
                    idx = gimme_random_subset_idx(len(self.X), lmax_max_size)
                    self._lmax_subset_X = self.X[idx]
                    self._lmax_subset_Y = self.Y[idx]
                else:
                    self._lmax_subset_X = self.X
                    self._lmax_subset_Y = self.Y
            X_subset = self._lmax_subset_X
            Y_subset = self._lmax_subset_Y

            preds = self.net(X_subset).squeeze(dim=-1)
            loss = self.loss_fn(preds, Y_subset)

            if self.eigenvector_cache is not None:
                max_iterations = 100 if not self.use_power_iteration else 1000
                tolerance = 0.005 if self.num_eigenvalues < 6 else 0.03

                eigenvalues, eigenvectors = compute_eigenvalues(
                    loss,
                    self.net,
                    k=self.num_eigenvalues,
                    max_iterations=max_iterations,
                    reltol=tolerance,
                    eigenvector_cache=self.eigenvector_cache,
                    return_eigenvectors=True,
                    use_power_iteration=self.use_power_iteration,
                )

                if self.num_eigenvalues == 1:
                    self.eigenvector_cache.store_eigenvector(eigenvectors, eigenvalues.item())
                    lmax_value = eigenvalues
                else:
                    self.eigenvector_cache.store_eigenvectors(
                        [eigenvectors[:, i] for i in range(eigenvectors.shape[1])],
                        eigenvalues.tolist(),
                    )
                    lmax_value = eigenvalues[0]
            else:
                eigenvalues = compute_eigenvalues(
                    loss,
                    self.net,
                    k=self.num_eigenvalues,
                    max_iterations=200,
                    reltol=0.03,
                    use_power_iteration=self.use_power_iteration,
                )
                if self.num_eigenvalues == 1:
                    lmax_value = eigenvalues
                else:
                    lmax_value = eigenvalues[0]

            if self.num_eigenvalues > 1:
                metrics['all_eigenvalues'] = eigenvalues
                if self.eigenvalues_file is not None:
                    eigenvalues_data = {
                        'epoch': epoch,
                        'step': step_number,
                        'eigenvalues': eigenvalues.tolist()
                        if isinstance(eigenvalues, torch.Tensor)
                        else [eigenvalues],
                    }
                    self.eigenvalues_log.append(eigenvalues_data)
                    if len(self.eigenvalues_log) > 1:
                        self.eigenvalues_file.write(',\n')
                    json.dump(eigenvalues_data, self.eigenvalues_file)
                    self.eigenvalues_file.flush()

            metrics['lmax'] = lmax_value.item()
            metrics['full_loss'] = loss.item()
            metrics['full_accuracy'] = calculate_accuracy(preds, Y_subset)

            epoch_loss_update = metrics['full_loss']

        # ----- Probe-batch GBS (after lmax so u_full is fresh from eigenvector cache) -----
        if 'probe_batch_gbs' in self.measurements:
            if frequency_calculator.should_measure('probe_batch_gbs', ctx):
                u_full = (self.eigenvector_cache.eigenvectors[0]
                          if self.eigenvector_cache and self.eigenvector_cache.eigenvectors
                          else None)
                lambda_max_cached = (self.eigenvector_cache.eigenvalues[0]
                                     if self.eigenvector_cache and self.eigenvector_cache.eigenvalues
                                     else None)
                save_dist = (self.compute_distributions and
                             frequency_calculator.should_measure('distributions', ctx))

                # Compute full-batch gradient every probe step (same subset cap as lmax)
                lmax_max_size = self._full_subset_cap()
                if len(self.X) > lmax_max_size:
                    idx = gimme_random_subset_idx(len(self.X), lmax_max_size)
                    X_gf = self.X[idx]
                    Y_gf = self.Y[idx]
                else:
                    X_gf, Y_gf = self.X, self.Y
                self.net.zero_grad()
                _preds_gf = self.net(X_gf).squeeze(dim=-1)
                _loss_gf = self.loss_fn(_preds_gf, Y_gf)
                _loss_gf.backward()
                grad_full = torch.cat([p.grad.flatten() for p in self.net.parameters()]).detach()
                self.net.zero_grad()

                scalars, distributions = compute_gbs_probe_batches(
                    self.net, self.X, self.Y, self.loss_fn, self.optimizer,
                    batch_size=self._measurement_batch_size(),
                    n_probe=self.probe_samples,
                    power_iters=self.power_iters,
                    u_full=u_full,
                    lambda_max=lambda_max_cached,
                    return_distributions=save_dist,
                    grad_full=grad_full,
                    compute_uB=self.compute_quantities_with_uB,
                )
                (A, B, GBS, SBS,
                 A_u, B_u, GBS_u,
                 A_g, B_g, GBS_g,
                 A_gfull, B_gfull, GBS_gfull,
                 A_ufull, B_ufull, GBS_ufull,
                 A_cos_sBgB, GBS_cos_sBgB,
                 A_cos_sBgfull, GBS_cos_sBgfull,
                 A_cos_gBgfull, GBS_cos_gBgfull) = scalars
                if save_dist and distributions is not None:
                    self._dist_records.append({'step': step_number, **distributions})
                metrics['A'] = A
                metrics['B'] = B
                metrics['GBS'] = GBS
                metrics['SBS'] = SBS
                metrics['A_u'] = A_u
                metrics['B_u'] = B_u
                metrics['GBS_u'] = GBS_u
                metrics['A_g'] = A_g
                metrics['B_g'] = B_g
                metrics['GBS_g'] = GBS_g
                metrics['A_gfull'] = A_gfull
                metrics['B_gfull'] = B_gfull
                metrics['GBS_gfull'] = GBS_gfull
                metrics['A_ufull'] = A_ufull
                metrics['B_ufull'] = B_ufull
                metrics['GBS_ufull'] = GBS_ufull
                metrics['A_cos_sBgB'] = A_cos_sBgB
                metrics['GBS_cos_sBgB'] = GBS_cos_sBgB
                metrics['A_cos_sBgfull'] = A_cos_sBgfull
                metrics['GBS_cos_sBgfull'] = GBS_cos_sBgfull
                metrics['A_cos_gBgfull'] = A_cos_gBgfull
                metrics['GBS_cos_gBgfull'] = GBS_cos_gBgfull

        if 'full_loss_warmup' in self.measurements:
            if frequency_calculator.should_measure('full_loss_warmup', ctx):
                with torch.no_grad():
                    full_preds = self.net(self.X).squeeze(dim=-1)
                    full_loss_tensor = self.loss_fn(full_preds, self.Y)
                    full_accuracy_value = calculate_accuracy(full_preds, self.Y)
                metrics['full_loss'] = float(full_loss_tensor.item())
                metrics['full_accuracy'] = float(full_accuracy_value)
                epoch_loss_update = metrics['full_loss']

        if 'full_loss' in self.measurements:
            if frequency_calculator.should_measure('full_loss', ctx):
                if np.isnan(metrics['full_loss']):
                    with torch.no_grad():
                        full_preds = self.net(self.X).squeeze(dim=-1)
                        full_loss_tensor = self.loss_fn(full_preds, self.Y)
                        metrics['full_loss'] = float(full_loss_tensor.item())
                        metrics['full_accuracy'] = float(calculate_accuracy(full_preds, self.Y))
                epoch_loss_update = metrics['full_loss']

        metrics['epoch_loss_update'] = epoch_loss_update

        return metrics



# -------------------------------------
# Section: Training Function
# -------------------------------------


def train(
            net,
            optimizer,
            data, # tuple of X_train, Y_train, X_test, Y_test
            max_epochs,
            max_steps,
            batch_size,
            save_to, #folder
            device,
            verbose=True,
            loss_fn=nn.MSELoss(),
            permute=True,
            data_ordering_seed=None,
            stop_loss=None,
            measurements: set = {},
            cache_eigenvectors: bool = True,
            use_power_iteration: bool = False,
            num_eigenvalues: int = 1,
            probe_samples: int = 128,
            power_iters: int = 50,
            compute_distributions: bool = False,
            compute_quantities_with_uB: bool = True,
            track_from: int | None = None,
            track_until: int | None = None,
            fixed_u: bool = False,
            projection_save_every: int = 100,
            track_stride: int = 1,
            lobpcg_max_iters: int = 20,
            lobpcg_reltol: float = 0.02,
            measurement_batch_size_cap: int | None = None,
            ):

    # -------------------------------------
    # Section: Setup
    # -------------------------------------
    start_time = time.time()
    print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    assert max_epochs is not None or max_steps is not None
    if max_epochs is None:
        max_epochs = 100000

    # ----- Dataset Wiring -----
    X_train, Y_train, X_test, Y_test = data

    X, Y = X_train, Y_train

    # ----- Device Alignment -----
    net = net.to(device)
    net.train()
    net.float()

    X = X.to(device)
    Y = Y.to(device)

    ordering_generator = None
    if data_ordering_seed is not None:
        data_ordering_seed = abs(int(data_ordering_seed))
        ordering_generator = torch.Generator(device=X.device)
        ordering_max_seed = 2**63 - 1

    # ----- Storage Preparation -----
    save_to.mkdir(parents=True, exist_ok=True)

    results_file = save_to / 'results.txt'
    if device == 'cpu':
        results_file = open(results_file, 'a', buffering=1)
        torch.set_num_threads(40)
    else:
        results_file = open(results_file, 'a', buffering=1_000)

    # ----- State Initialization -----
    step_number = -1
    steps_since_restart = -1

    # ----- Print active measurements -----
    measurement_names = {
        'lmax': 'lambda_max',
        'batch_sharpness': 'batch_sharpness',
        'probe_batch_gbs': 'probe_batch_gbs',
        'step_sharpness': 'step_sharpness',
        'full_loss': 'full_loss',
        'full_loss_warmup': 'full_loss_warmup',
        'final': 'final',
    }
    freq_keys = {
        'lmax': 'full_batch_lambda_max',
        'batch_sharpness': 'batch_sharpness',
        'probe_batch_gbs': 'probe_batch_gbs',
        'step_sharpness': 'step_sharpness',
        'full_loss': 'full_loss',
    }
    active = []
    for m in measurement_names:
        if m in measurements:
            fkey = freq_keys.get(m)
            interval = frequency_calculator._intervals.get(fkey) if fkey else None
            if interval is not None:
                active.append(f"{measurement_names[m]} (every {interval})")
            else:
                active.append(f"{measurement_names[m]} (adaptive)")
    if active:
        print(f"Logging: {', '.join(active)}")
    else:
        print("No measurements enabled")

    # ----- Training State Trackers -----
    epoch_loss = float('+inf')
    stop_training = False
    completed_epoch = -1

    # ----- Eigenvector Cache Setup -----
    eigenvector_cache = None
    if (('lmax' in measurements or 'final' in measurements)) and cache_eigenvectors:
        max_cache = 5
        if num_eigenvalues is not None:
            max_cache = max(max_cache, num_eigenvalues)
        eigenvector_cache = EigenvectorCache(max_eigenvectors=max_cache)

    # ----- Distributions File -----
    distributions_file = save_to / 'distributions.npz'

    # ----- Measurement Runner Wiring -----
    measurement_runner = MeasurementRunner(
        net=net,
        optimizer=optimizer,
        loss_fn=loss_fn,
        full_inputs=(X, Y),
        measurements=measurements,
        device=device,
        batch_size=batch_size,
        save_dir=save_to,
        eigenvector_cache=eigenvector_cache,
        num_eigenvalues=num_eigenvalues,
        use_power_iteration=use_power_iteration,
        probe_samples=probe_samples,
        power_iters=power_iters,
        compute_distributions=compute_distributions,
        distributions_file=distributions_file,
        compute_quantities_with_uB=compute_quantities_with_uB,
        measurement_batch_size_cap=measurement_batch_size_cap,
    )

    # ----- Projection Tracker -----
    projection_tracker = None
    if track_from is not None and track_until is not None:
        projection_tracker = ProjectionTracker(
            net=net,
            X=X,
            Y=Y,
            loss_fn=loss_fn,
            track_from=track_from,
            track_until=track_until,
            save_dir=save_to,
            device=device,
            optimizer_wrapper=optimizer,
            fixed_u=fixed_u,
            save_every=projection_save_every,
            track_stride=track_stride,
            lobpcg_max_iters=lobpcg_max_iters,
            lobpcg_reltol=lobpcg_reltol,
        )
        print(f"Projection tracking enabled for steps {track_from}–{track_until}"
              + (" [fixed_u]" if fixed_u else "")
              + f" [stride={track_stride} save_every={projection_save_every}"
              + f" lobpcg_max_iters={lobpcg_max_iters} lobpcg_reltol={lobpcg_reltol}]")

    # -------------------------------------
    # Section: Training Step
    # -------------------------------------
    initial_step = step_number + 1
    pbar = tqdm(total=max_steps - max(initial_step, 0), initial=0,
                desc="Training", unit="step", dynamic_ncols=True,
                file=sys.stderr)

    for epoch in range(0, max_epochs):
        completed_epoch = epoch

        if step_number >= max_steps:
            break

        # --- Epoch Data Preparation ---
        if permute:
            if ordering_generator is not None:
                epoch_seed = (data_ordering_seed + epoch) % ordering_max_seed
                if epoch_seed == 0:
                    epoch_seed = ordering_max_seed
                ordering_generator.manual_seed(epoch_seed)
                shuffle = T.randperm(len(X), generator=ordering_generator, device=X.device)
            else:
                shuffle = T.randperm(len(X), device=X.device)
            X_shuffled = X[shuffle]
            Y_shuffled = Y[shuffle]
        else:
            X_shuffled = X
            Y_shuffled = Y

        losses_in_epoch = []
        if stop_training:
            break

        # --- Minibatch Iteration ---
        location_within_epoch = 0
        for i in range(location_within_epoch, len(X) // batch_size):
            step_number += 1
            steps_since_restart += 1

            if step_number >= max_steps:
                stop_training = True
                break

            msg = f"{epoch},{step_number},"
            # --- Measurement Context and Sampling ---
            ctx = MeasurementContext(
                step_number=step_number,
                batch_size=batch_size,
                epoch=epoch,
                device=str(device),
                lr=optimizer.param_groups[0]['lr'],
                steps_since_restart=steps_since_restart,
            )

            X_batch = X_shuffled[i*batch_size : (i+1)*batch_size]
            Y_batch = Y_shuffled[i*batch_size : (i+1)*batch_size]

            # -------------------------------------
            # Section: Measurements
            # -------------------------------------
            metrics = measurement_runner.collect(
                ctx=ctx,
                optimizer=optimizer,
                X_batch=X_batch,
                Y_batch=Y_batch,
                epoch=epoch,
                step_in_epoch=i,
                step_number=step_number,
            )

            # --- Epoch-Level Loss Tracking ---
            if metrics['epoch_loss_update'] is not None:
                if math.isnan(metrics['epoch_loss_update']):
                    pbar.close()
                    print('Full loss is NaN, the network prolly diverged, stopping the training')
                    results_file.flush()
                    results_file.close()
                    measurement_runner.close()
                    return
                epoch_loss = metrics['epoch_loss_update']

            if stop_loss is not None and epoch_loss < stop_loss:
                stop_training = True
                break

            # -------------------------------------
            # Section: Training Step (Update)
            # -------------------------------------
            optimizer.zero_grad()

            # Standard SGD step
            preds = net(X_batch).squeeze(dim=-1)

            loss = loss_fn(preds, Y_batch)

            if math.isinf(loss.item()) or math.isnan(loss.item()):
                pbar.close()
                results_file.flush()
                results_file.close()
                measurement_runner.close()
                raise ValueError("Loss is inf or NaN, stopping the training")

            loss.backward()

            # --- Projection Tracking (pre-step) ---
            _theta_before = None
            if projection_tracker is not None and projection_tracker.should_track(step_number):
                _theta_before = projection_tracker.pre_step(step_number, X_batch, Y_batch, loss)

            optimizer.step()

            # --- Projection Tracking (post-step: record Δθ) ---
            if projection_tracker is not None and _theta_before is not None:
                projection_tracker.post_step(step_number, _theta_before)

            # Handle loss value
            batch_loss = loss.item()
            losses_in_epoch.append(batch_loss)

            # -------------------------------------
            # Section: Logging (Step)
            # -------------------------------------
            msg += (
                f"{batch_loss},{metrics['full_loss']},"
                f"{metrics['lmax']},{metrics['step_sharpness']},"
                f"{metrics['batch_sharpness']},"
                f"{metrics['A']},{metrics['B']},{metrics['GBS']},{metrics['SBS']},"
                f"{metrics['A_u']},{metrics['B_u']},{metrics['GBS_u']},"
                f"{metrics['A_g']},{metrics['B_g']},{metrics['GBS_g']},"
                f"{metrics['A_gfull']},{metrics['B_gfull']},{metrics['GBS_gfull']},"
                f"{metrics['A_ufull']},{metrics['B_ufull']},{metrics['GBS_ufull']},"
                f"{metrics['A_cos_sBgB']},{metrics['GBS_cos_sBgB']},"
                f"{metrics['A_cos_sBgfull']},{metrics['GBS_cos_sBgfull']},"
                f"{metrics['A_cos_gBgfull']},{metrics['GBS_cos_gBgfull']},"
                f"{metrics['full_accuracy']}"
            )
            results_file.write(msg + "\n")

            pbar.update(1)

            if step_number > 0 and step_number % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"[step {step_number}/{max_steps}] elapsed "
                      f"{time.strftime('%H:%M:%S', time.gmtime(elapsed))}",
                      flush=True)


        # --- Epoch Finalization ---
        epoch_loss = np.mean(losses_in_epoch)

        results_file.flush()

    pbar.close()


    results_file.close()

    if projection_tracker is not None:
        projection_tracker.save()

    measurement_runner.close()

    if 'final' in measurements:
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())

        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

        optimizer.zero_grad()

        lmax_max_size = 8192
        if str(device).startswith('cuda'):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            if total_memory < 20 * 1024**3:
                if isinstance(net, CNN):
                    lmax_max_size = 2048 + 512
                if isinstance(net, ResNet):
                    lmax_max_size = 512

        if len(X) > lmax_max_size:
            print(
                f"Final lambda_max: using subset of {lmax_max_size} samples "
                f"instead of full dataset ({len(X)} samples)"
            )
            idx = gimme_random_subset_idx(len(X), lmax_max_size)
            X_subset = X[idx]
            Y_subset = Y[idx]
        else:
            X_subset = X
            Y_subset = Y

        extra_steps = 500
        lambda_max_measurements = []
        eigenvalues_list = []

        measurements_to_run = 3

        for measurement_idx in range(measurements_to_run):
            was_training = net.training
            net.eval()
            try:
                preds = net(X_subset).squeeze(dim=-1)
                loss = loss_fn(preds, Y_subset)
                if num_eigenvalues == 1:
                    eigenvalue = compute_eigenvalues(
                        loss,
                        net,
                        k=num_eigenvalues,
                        max_iterations=200,
                        reltol=0.01,
                        eigenvector_cache=eigenvector_cache,
                        use_power_iteration=use_power_iteration,
                    )

                    lambda_val = float(eigenvalue.item())
                    if measurement_idx == measurements_to_run - 1:
                        eigenvalues_list = [lambda_val]
                else:
                    raise NotImplementedError("Final measurement for multiple eigenvalues not implemented yet")

                lambda_max_measurements.append(lambda_val)
                full_loss_value = float(loss.item())
                full_accuracy_value = float(calculate_accuracy(preds, Y_subset))
            finally:
                net.train(was_training)

            if measurement_idx == 0:
                for _ in range(extra_steps):
                    step_number += 1
                    batch_indices = torch.randint(len(X), (batch_size,), device=device)
                    X_batch = X[batch_indices]
                    Y_batch = Y[batch_indices]
                    optimizer.zero_grad()
                    preds_extra = net(X_batch).squeeze(dim=-1)
                    extra_loss = loss_fn(preds_extra, Y_batch)
                    if math.isinf(extra_loss.item()) or math.isnan(extra_loss.item()):
                        raise ValueError("Loss is inf or NaN during final extra steps, stopping the training")
                    extra_loss.backward()
                    optimizer.step()
                optimizer.zero_grad()

        lambda_max_value = sum(lambda_max_measurements) / len(lambda_max_measurements)

        final_step_number = max(step_number, 0)
        final_epoch_index = max(completed_epoch, 0)

        final_metrics = {
            "lambda_max": lambda_max_value,
            "lambda_maxes": lambda_max_measurements,
            "eigenvalues": eigenvalues_list,
            "full_loss": full_loss_value,
            "full_accuracy": full_accuracy_value,
            "step": final_step_number,
            "epoch": final_epoch_index,
            "dataset_size": int(len(X)),
            "subset_size": int(len(X_subset)),
            "num_eigenvalues": int(num_eigenvalues),
            "use_power_iteration": bool(use_power_iteration),
            "timestamp": timestamp,
        }

        final_json_path = save_to / 'final.json'
        write_json_atomic(final_json_path, final_metrics)
        print(f"Final lambda max measurements = {lambda_max_measurements}, avg = {lambda_max_value:.6f}")
        print(f"Final loss = {full_loss_value:.6f}, accuracy = {full_accuracy_value:.4f}")
        print(f"Final metrics saved to {final_json_path}")
    # ----- Final Reporting -----
    end_time = time.time()
    print(f"Training finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total training time: {end_time - start_time:.2f} seconds")
