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
from utils.wandb_utils import (
    save_checkpoint_wandb,
    generate_run_id,
)
from utils.measure import (
    EigenvectorCache,
    param_length,
    gimme_random_subset_idx,
    compute_eigenvalues,
    compute_grad_H_grad,
    calculate_averaged_grad_H_grad_step,
    compute_gbs_actual_batch,
    compute_gbs_probe_batches,
    compute_gbs_full_batch,
    calculate_accuracy,
)
from utils.frequency import frequency_calculator, MeasurementContext


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
        step_to_start,
        run_id,
        probe_samples,
        gbs_power_iters,
        compute_u=True,
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
        self.step_to_start = step_to_start
        self.run_id = run_id
        self.probe_samples = probe_samples
        self.gbs_power_iters = gbs_power_iters
        self.compute_u = compute_u

        self.eigenvalues_log = []
        if 'lmax' in measurements and num_eigenvalues > 1:
            eigenvalues_path = save_dir / 'eigenvalues.json'
            self.eigenvalues_file = open(eigenvalues_path, 'w')
            self.eigenvalues_file.write('[\n')
        else:
            self.eigenvalues_file = None

    def close(self):
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
            'A_actual': np.nan,
            'A_u_actual': np.nan,
            'B_actual': np.nan,
            'B_u_actual': np.nan,
            'out_actual': np.nan,
            'out_actual_u': np.nan,
            'A_probe': np.nan,
            'A_u_probe': np.nan,
            'B_probe': np.nan,
            'B_u_probe': np.nan,
            'out_probe': np.nan,
            'out_probe_u': np.nan,
            'A_full': np.nan,
            'A_u_full': np.nan,
            'B_full': np.nan,
            'B_u_full': np.nan,
            'out_full': np.nan,
            'out_full_u': np.nan,
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
                    batch_size=self.batch_size,
                    n_estimates=self.probe_samples,
                    min_estimates=20,
                    eps=0.005,
                )

        # ----- Actual-batch GBS -----
        if 'actual_batch_gbs' in self.measurements:
            if frequency_calculator.should_measure('actual_batch_gbs', ctx):
                A, A_u, B, B_u, out_actual, out_actual_u = compute_gbs_actual_batch(
                    self.net, X_batch, Y_batch, self.loss_fn, self.optimizer,
                    power_iters=self.gbs_power_iters,
                    compute_u=self.compute_u,
                )
                metrics['A_actual'] = A
                metrics['A_u_actual'] = A_u
                metrics['B_actual'] = B
                metrics['B_u_actual'] = B_u
                metrics['out_actual'] = out_actual
                metrics['out_actual_u'] = out_actual_u

        # ----- Probe-batch GBS -----
        if 'probe_batch_gbs' in self.measurements:
            if frequency_calculator.should_measure('probe_batch_gbs', ctx):
                A, A_u, B, B_u, out_probe, out_probe_u = compute_gbs_probe_batches(
                    self.net, self.X, self.Y, self.loss_fn, self.optimizer,
                    batch_size=self.batch_size,
                    n_probe=self.probe_samples,
                    power_iters=self.gbs_power_iters,
                    compute_u=self.compute_u,
                )
                metrics['A_probe'] = A
                metrics['A_u_probe'] = A_u
                metrics['B_probe'] = B
                metrics['B_u_probe'] = B_u
                metrics['out_probe'] = out_probe
                metrics['out_probe_u'] = out_probe_u

        # ----- Instantaneous step sharpness (current-batch Rayleigh quotient) -----
        if 'step_sharpness' in self.measurements:
            if frequency_calculator.should_measure('step_sharpness', ctx):
                self.net.zero_grad()
                preds = self.net(X_batch).squeeze(dim=-1)
                loss = self.loss_fn(preds, Y_batch)
                metrics['step_sharpness'] = compute_grad_H_grad(loss, self.net).item()

        # ----- Eigenvalues/Lambda max (full batch) -----
        lmax_now = False
        if 'lmax' in self.measurements:
            measurement_type = 'full_batch_lambda_max'
            lmax_now = frequency_calculator.should_measure(measurement_type, ctx)

        if lmax_now:
            if str(self.device).startswith('cuda'):
                torch.cuda.empty_cache()
            optimizer.zero_grad()

            lmax_max_size = 4096
            if str(self.device).startswith('cuda'):
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if total_memory < 20 * 1024**3:
                    if isinstance(self.net, CNN):
                        lmax_max_size = 2048 + 512
                    if isinstance(self.net, ResNet):
                        lmax_max_size = 512
                    if isinstance(self.net, WideResNet) or isinstance(self.net, WideResNetNoBN):
                        lmax_max_size = 1024

            if len(self.X) > lmax_max_size:
                idx = gimme_random_subset_idx(len(self.X), lmax_max_size)
                X_subset = self.X[idx]
                Y_subset = self.Y[idx]
            else:
                X_subset = self.X
                Y_subset = self.Y

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

            # ----- Full-batch GBS (reuses X_subset, Y_subset, and u from eigenvector cache) -----
            if 'full_batch_gbs' in self.measurements and self.eigenvector_cache is not None:
                u = self.eigenvector_cache.eigenvectors[0]
                A_f, A_u_f, B_f, B_u_f, out_f, out_f_u = compute_gbs_full_batch(
                    self.net, X_subset, Y_subset, X_batch, Y_batch,
                    self.loss_fn, self.optimizer, u,
                    compute_u=self.compute_u,
                )
                metrics['A_full'] = A_f
                metrics['A_u_full'] = A_u_f
                metrics['B_full'] = B_f
                metrics['B_u_full'] = B_u_f
                metrics['out_full'] = out_f
                metrics['out_full_u'] = out_f_u

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
            epoch_to_start=0,
            step_to_start=0,
            measurements: set = {},
            cache_eigenvectors: bool = True,
            use_power_iteration: bool = False,
            num_eigenvalues: int = 1,
            checkpoint_every_n_steps: int = None,
            run_id: str = None,
            probe_samples: int = 128,
            gbs_power_iters: int = 50,
            compute_u: bool = True,
            ):

    # -------------------------------------
    # Section: Setup
    # -------------------------------------
    start_time = time.time()
    print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    NET_SAVES_PER_TRAINING = 100

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

    model_save_path = save_to / 'checkpoints'

    results_file = save_to / 'results.txt'
    if device == 'cpu':
        results_file = open(results_file, 'a', buffering=1)
        torch.set_num_threads(40)
    else:
        results_file = open(results_file, 'a', buffering=1_000)

    # ----- State Initialization -----
    step_number = -1 if step_to_start == 0 else step_to_start
    steps_since_restart = -1

    # ----- Checkpoint Interval Selection -----
    if checkpoint_every_n_steps is None:
        checkpoint_every_n_steps = max(max_steps // NET_SAVES_PER_TRAINING, 1)
    print(f"Will save checkpoints every {checkpoint_every_n_steps} steps")

    # ----- Print active measurements -----
    measurement_names = {
        'lmax': 'lambda_max',
        'batch_sharpness': 'batch_sharpness',
        'actual_batch_gbs': 'actual_batch_gbs',
        'probe_batch_gbs': 'probe_batch_gbs',
        'full_batch_gbs': 'full_batch_gbs',
        'step_sharpness': 'step_sharpness',
        'full_loss': 'full_loss',
        'full_loss_warmup': 'full_loss_warmup',
        'final': 'final',
    }
    freq_keys = {
        'lmax': 'full_batch_lambda_max',
        'batch_sharpness': 'batch_sharpness',
        'actual_batch_gbs': 'actual_batch_gbs',
        'probe_batch_gbs': 'probe_batch_gbs',
        'full_batch_gbs': 'full_batch_lambda_max',  # fires at same time as lmax
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
    completed_epoch = epoch_to_start - 1

    # ----- Eigenvector Cache Setup -----
    eigenvector_cache = None
    if (('lmax' in measurements or 'final' in measurements)) and cache_eigenvectors:
        max_cache = 5
        if num_eigenvalues is not None:
            max_cache = max(max_cache, num_eigenvalues)
        eigenvector_cache = EigenvectorCache(max_eigenvectors=max_cache)

    # ----- Run Identification -----
    run_id = run_id or generate_run_id()

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
        step_to_start=step_to_start,
        run_id=run_id,
        probe_samples=probe_samples,
        gbs_power_iters=gbs_power_iters,
        compute_u=compute_u,
    )

    # -------------------------------------
    # Section: Training Step
    # -------------------------------------
    initial_step = step_number + 1
    pbar = tqdm(total=max_steps - max(initial_step, 0), initial=0,
                desc="Training", unit="step", dynamic_ncols=True,
                file=sys.stderr)

    def _status(text):
        pbar.set_postfix_str(text, refresh=False)

    for epoch in range(epoch_to_start, max_epochs):
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

        # initialize the correct starting point for the batch picking
        location_within_epoch = 0
        if epoch == epoch_to_start:
            steps_per_epoch = len(X) // batch_size
            if step_to_start > 0:
                location_within_epoch = step_to_start % steps_per_epoch

        # --- Minibatch Iteration ---
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
                _status(f"Loss {epoch_loss} below stop_loss {stop_loss}, stopping")
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
                raise ValueError("Loss is inf or NaN, stopping the training")

            loss.backward()

            optimizer.step()

            # Handle loss value
            batch_loss = loss.item()
            losses_in_epoch.append(batch_loss)

            # --- Checkpoint Handling ---
            checkpoint_path = save_checkpoint_wandb(
                model=net,
                optimizer=optimizer,
                step=step_number,
                epoch=epoch,
                loss=batch_loss,
                run_id=run_id,
                save_every_n_steps=checkpoint_every_n_steps
            )

            # -------------------------------------
            # Section: Logging (Step)
            # -------------------------------------
            msg += (
                f"{batch_loss},{metrics['full_loss']},"
                f"{metrics['lmax']},{metrics['step_sharpness']},"
                f"{metrics['batch_sharpness']},"
                f"{metrics['A_actual']},{metrics['A_u_actual']},{metrics['B_actual']},{metrics['B_u_actual']},{metrics['out_actual']},{metrics['out_actual_u']},"
                f"{metrics['A_probe']},{metrics['A_u_probe']},{metrics['B_probe']},{metrics['B_u_probe']},{metrics['out_probe']},{metrics['out_probe_u']},"
                f"{metrics['A_full']},{metrics['A_u_full']},{metrics['B_full']},{metrics['B_u_full']},{metrics['out_full']},{metrics['out_full_u']},"
                f"{metrics['full_accuracy']}"
            )
            results_file.write(msg + "\n")

            _status(f"ep {epoch}  step {step_number}  loss={batch_loss:.4f}")

            pbar.update(1)


        # --- Epoch Finalization ---
        epoch_loss = np.mean(losses_in_epoch)

        results_file.flush()

    pbar.close()


    # -------------------------------------
    # Section: Logging
    # -------------------------------------
    # ----- Final Checkpoint Save -----
    final_checkpoint_path = save_checkpoint_wandb(
        model=net,
        optimizer=optimizer,
        step=step_number,
        epoch=epoch,
        loss=batch_loss,
        run_id=run_id,
        save_every_n_steps=1
    )
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    results_file.close()

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
            "run_id": run_id,
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
