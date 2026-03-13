import torch
import torch as T


class OptimizerWrapper:
    """Base class wrapping a torch optimizer + step direction computation."""

    name: str

    def __init__(self, inner):
        self.inner = inner

    def step(self):
        self.inner.step()

    def zero_grad(self):
        self.inner.zero_grad()

    @property
    def param_groups(self):
        return self.inner.param_groups

    @property
    def state(self):
        return self.inner.state

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state_dict):
        self.inner.load_state_dict(state_dict)

    def compute_step_direction(self, grads_flat, params):
        """Return the detached step direction s_b given current grads.
        Must be implemented by subclasses."""
        raise NotImplementedError


class SGDOptimizer(OptimizerWrapper):
    name = "SGD"

    def __init__(self, net, lr):
        inner = T.optim.SGD(net.parameters(), lr=lr, momentum=0)
        super().__init__(inner)

    def compute_step_direction(self, grads_flat, params):
        lr = self.param_groups[0]['lr']
        return (-lr * grads_flat).detach()


class SGDMomentumOptimizer(OptimizerWrapper):
    name = "SGD-Momentum"

    def __init__(self, net, lr, beta=0.9):
        self.beta = beta
        inner = T.optim.SGD(net.parameters(), lr=lr, momentum=beta)
        super().__init__(inner)

    def compute_step_direction(self, grads_flat, params):
        lr = self.param_groups[0]['lr']
        beta = self.beta

        # Build the momentum-aware update: s = -lr * (g + beta * v)
        # where v is the momentum buffer from the optimizer state
        offset = 0
        pieces = []
        for p in params:
            length = p.numel()
            g_p = grads_flat[offset:offset + length].view(p.shape)

            state = self.inner.state.get(p)
            if state and 'momentum_buffer' in state:
                v_p = state['momentum_buffer']
                pieces.append((g_p + beta * v_p).view(-1))
            else:
                # First step: no buffer yet, direction is just the gradient
                pieces.append(g_p.view(-1))

            offset += length

        return (-lr * torch.cat(pieces)).detach()


class SGDNesterovOptimizer(OptimizerWrapper):
    name = "SGD-Nesterov"

    def __init__(self, net, lr, beta=0.9):
        self.beta = beta
        inner = T.optim.SGD(net.parameters(), lr=lr, momentum=beta, nesterov=True)
        super().__init__(inner)

    def compute_step_direction(self, grads_flat, params):
        lr = self.param_groups[0]['lr']
        beta = self.beta

        # Nesterov: buf_new = beta * buf + g, then s = -lr * (g + beta * buf_new)
        offset = 0
        pieces = []
        for p in params:
            length = p.numel()
            g_p = grads_flat[offset:offset + length].view(p.shape)

            state = self.inner.state.get(p)
            if state and 'momentum_buffer' in state:
                v_p = state['momentum_buffer']
                buf_new = beta * v_p + g_p
                pieces.append((g_p + beta * buf_new).view(-1))
            else:
                pieces.append(g_p.view(-1))

            offset += length

        return (-lr * torch.cat(pieces)).detach()


class AdamOptimizer(OptimizerWrapper):
    name = "Adam"

    def __init__(self, net, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        inner = T.optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
        super().__init__(inner)

    def compute_step_direction(self, grads_flat, params):
        lr = self.param_groups[0]['lr']
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps

        # Hypothetical Adam step: use current state (m, v, t) with fresh grad g
        offset = 0
        pieces = []
        for p in params:
            length = p.numel()
            g_p = grads_flat[offset:offset + length].view(p.shape)

            state = self.inner.state.get(p)
            if state and 'exp_avg' in state:
                m = state['exp_avg']
                v = state['exp_avg_sq']
                t = state['step'].item()

                m_new = beta1 * m + (1 - beta1) * g_p
                v_new = beta2 * v + (1 - beta2) * g_p * g_p
                t_new = t + 1

                bc1 = 1 - beta1 ** t_new
                bc2 = 1 - beta2 ** t_new

                m_hat = m_new / bc1
                denom = (v_new / bc2).sqrt() + eps
                pieces.append((-lr * m_hat / denom).view(-1))
            else:
                # First step: m=0, v=0, t=0
                m_new = (1 - beta1) * g_p
                v_new = (1 - beta2) * g_p * g_p
                bc1 = 1 - beta1
                bc2 = 1 - beta2
                m_hat = m_new / bc1  # = g_p
                denom = (v_new / bc2).sqrt() + eps  # = |g_p| + eps
                pieces.append((-lr * m_hat / denom).view(-1))

            offset += length

        return torch.cat(pieces).detach()


def _newton_schulz5(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration: maps G to a matrix with ~orthonormal rows/cols."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float() / (G.norm() + eps)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    # Scale so that update magnitude is consistent regardless of matrix shape
    scale = max(1.0, (G.shape[0] / G.shape[1]) ** 0.5)
    return (X * scale).to(G.dtype)


class MuonOptimizer(OptimizerWrapper):
    """Muon optimizer (Keller Jordan variant).

    2D+ parameters: SGD momentum + Newton-Schulz orthogonalization.
    1D parameters (biases etc.): Adam with a separate lr.
    """
    name = "Muon"

    def __init__(self, net, lr, momentum=0.95, nesterov=True, ns_steps=5,
                 adam_lr=3e-4, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8):
        self.lr       = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.adam_lr   = adam_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps   = adam_eps

        all_params = list(net.parameters())
        self._muon_params = [p for p in all_params if p.ndim >= 2]
        self._adam_params  = [p for p in all_params if p.ndim < 2]

        # Momentum buffers for Muon params (managed manually)
        self._bufs = {id(p): torch.zeros_like(p) for p in self._muon_params}

        # Adam sub-optimizer for 1D params
        self._adam = (
            T.optim.Adam(self._adam_params, lr=adam_lr,
                         betas=(adam_beta1, adam_beta2), eps=adam_eps)
            if self._adam_params else None
        )

        # Expose a fake inner so param_groups[0]['lr'] works for training.py logging
        inner = T.optim.SGD(self._muon_params if self._muon_params else all_params,
                            lr=lr, momentum=0)
        super().__init__(inner)

    # ------------------------------------------------------------------
    def zero_grad(self):
        for p in self._muon_params + self._adam_params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def _get_buf(self, p):
        """Return momentum buffer on the same device as p, migrating if needed."""
        buf = self._bufs[id(p)]
        if buf.device != p.device:
            buf = buf.to(p.device)
            self._bufs[id(p)] = buf
        return buf

    def step(self):
        # --- Muon step for 2D+ params ---
        for p in self._muon_params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            buf = self._get_buf(p)
            buf.mul_(self.momentum).add_(g)
            if self.nesterov:
                effective_g = g + self.momentum * buf
            else:
                effective_g = buf
            # Reshape to 2D, orthogonalize, reshape back
            orig_shape = effective_g.shape
            g2d = effective_g.reshape(orig_shape[0], -1)
            g_orth = _newton_schulz5(g2d, steps=self.ns_steps)
            p.data.add_(g_orth.reshape(orig_shape), alpha=-self.lr)

        # --- Adam step for 1D params ---
        if self._adam is not None:
            self._adam.step()

    # ------------------------------------------------------------------
    def compute_step_direction(self, grads_flat, params):
        """Simulate the Muon/Adam step given current gradients, without modifying state."""
        pieces = []
        offset = 0

        for p in params:
            length = p.numel()
            g_p = grads_flat[offset:offset + length].view(p.shape)
            offset += length

            if p.ndim >= 2:
                # Simulate Muon: hypothetical next momentum buffer
                buf = self._bufs.get(id(p))
                if buf is not None:
                    buf = buf.to(g_p.device)
                    buf_new = self.momentum * buf + g_p
                else:
                    buf_new = g_p
                if self.nesterov:
                    effective_g = g_p + self.momentum * buf_new
                else:
                    effective_g = buf_new

                orig_shape = effective_g.shape
                g2d = effective_g.reshape(orig_shape[0], -1)
                g_orth = _newton_schulz5(g2d, steps=self.ns_steps)
                pieces.append((-self.lr * g_orth.reshape(orig_shape)).reshape(-1))
            else:
                # Simulate Adam for 1D params
                state = self._adam.state.get(p) if self._adam else None
                if state and 'exp_avg' in state:
                    m = state['exp_avg']
                    v = state['exp_avg_sq']
                    t = state['step'].item() if torch.is_tensor(state['step']) else state['step']
                    m_new = self.adam_beta1 * m + (1 - self.adam_beta1) * g_p
                    v_new = self.adam_beta2 * v + (1 - self.adam_beta2) * g_p * g_p
                    t_new = t + 1
                    m_hat = m_new / (1 - self.adam_beta1 ** t_new)
                    denom = (v_new / (1 - self.adam_beta2 ** t_new)).sqrt() + self.adam_eps
                    pieces.append((-self.adam_lr * m_hat / denom).reshape(-1))
                else:
                    # First step
                    m_hat = g_p  # beta-corrected → just g at t=1
                    denom = g_p.abs() + self.adam_eps
                    pieces.append((-self.adam_lr * m_hat / denom).reshape(-1))

        return torch.cat(pieces).detach()


_REGISTRY = {
    'SGD': SGDOptimizer,
    'SGD-Momentum': SGDMomentumOptimizer,
    'SGD-Nesterov': SGDNesterovOptimizer,
    'Adam': AdamOptimizer,
    'Muon': MuonOptimizer,
}


def create_optimizer(name, net, lr, optimizer_params=None):
    """Factory: create an OptimizerWrapper by name.

    optimizer_params is a dict read straight from config, e.g. {'beta': 0.9}.
    """
    if optimizer_params is None:
        optimizer_params = {}

    if name not in _REGISTRY:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(_REGISTRY.keys())}")

    cls = _REGISTRY[name]
    return cls(net, lr, **optimizer_params)
