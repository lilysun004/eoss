import torch as T
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import math
import time
import sys
from tqdm import tqdm

from utils.jacobian import JacobianTracker


# -------------------------------------
# Section: Training Function
# -------------------------------------

def train(
    net,
    optimizer,
    data,
    max_epochs,
    max_steps,
    batch_size,
    save_to,
    device,
    loss_fn=None,
    data_ordering_seed=None,
    stop_loss=None,
    ema_taus=(1, 10, 100),
    full_loss_every=256,
):
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    start_time = time.time()
    print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"JacobianTracker taus={ema_taus}, full_loss every {full_loss_every} steps")

    assert max_epochs is not None or max_steps is not None
    if max_epochs is None:
        max_epochs = 100_000

    X_train, Y_train, X_test, Y_test = data
    X, Y = X_train, Y_train

    net = net.to(device)
    net.train()
    net.float()

    X = X.to(device)
    Y = Y.to(device)

    ordering_generator = None
    if data_ordering_seed is not None:
        data_ordering_seed = abs(int(data_ordering_seed))
        ordering_generator = torch.Generator(device=X.device)
        ordering_max_seed = 2 ** 63 - 1

    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)

    results_file = open(save_to / 'results.txt', 'a', buffering=1)

    tracker = JacobianTracker(net, taus=ema_taus)

    step_number = 0
    stop_training = False

    pbar = tqdm(
        total=max_steps, initial=0, desc="Training",
        unit="step", dynamic_ncols=True, file=sys.stderr
    )

    for epoch in range(max_epochs):
        if step_number >= max_steps or stop_training:
            break

        # --- Epoch shuffle ---
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
        losses_in_epoch = []

        for i in range(len(X) // batch_size):
            if step_number >= max_steps:
                stop_training = True
                break

            X_batch = X_shuffled[i * batch_size:(i + 1) * batch_size]
            Y_batch = Y_shuffled[i * batch_size:(i + 1) * batch_size]

            # --- JVP step (before update, same batch) ---
            jvp_metrics = tracker.step(net, X_batch, Y_batch, loss_fn, optimizer)

            # --- Full loss (periodic) ---
            full_loss = np.nan
            if step_number % full_loss_every == 0:
                with torch.no_grad():
                    full_preds = net(X).squeeze(dim=-1)
                    full_loss = loss_fn(full_preds, Y).item()

            # --- Training step ---
            optimizer.zero_grad()
            preds = net(X_batch).squeeze(dim=-1)
            loss = loss_fn(preds, Y_batch)

            if math.isinf(loss.item()) or math.isnan(loss.item()):
                results_file.flush()
                results_file.close()
                pbar.close()
                raise ValueError("Loss is inf or NaN, stopping training")

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            losses_in_epoch.append(batch_loss)

            # --- Accuracy (batch) ---
            with torch.no_grad():
                if preds.dim() > 1 and preds.shape[-1] > 1:
                    pred_cls = preds.argmax(dim=-1)
                    true_cls = Y_batch.argmax(dim=-1) if Y_batch.dim() > 1 else Y_batch
                    acc = (pred_cls == true_cls).float().mean().item()
                else:
                    acc = np.nan

            # --- Log row ---
            ell = jvp_metrics['ell']
            rho_vals = [jvp_metrics.get(f'rho_{tau}', np.nan) for tau in ema_taus]
            rho_str = ','.join(str(v) for v in rho_vals)
            row = (
                f"{epoch},{step_number},"
                f"{batch_loss},{full_loss},"
                f"{ell},{rho_str},"
                f"{acc}"
            )
            results_file.write(row + "\n")

            pbar.set_postfix_str(
                f"ep={epoch} loss={batch_loss:.4f} "
                f"ell={ell:.3f} rho_10={jvp_metrics.get('rho_10', float('nan')):.3f}",
                refresh=False,
            )
            pbar.update(1)
            step_number += 1

            if stop_loss is not None and losses_in_epoch:
                if np.mean(losses_in_epoch) < stop_loss:
                    stop_training = True
                    break

        results_file.flush()

    pbar.close()
    results_file.close()

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f}s")
