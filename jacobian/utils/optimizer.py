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

    @property
    def lr(self):
        return self.inner.param_groups[0]['lr']

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state_dict):
        self.inner.load_state_dict(state_dict)

    def get_diagonal_preconditioner(self):
        """Return flat diagonal preconditioner tensor, or None for SGD (P = I)."""
        return None


class SGDOptimizer(OptimizerWrapper):
    name = "SGD"

    def __init__(self, net, lr):
        inner = T.optim.SGD(net.parameters(), lr=lr, momentum=0)
        super().__init__(inner)

    def get_diagonal_preconditioner(self):
        return None


class SGDMomentumOptimizer(OptimizerWrapper):
    name = "SGD-Momentum"

    def __init__(self, net, lr, beta=0.9):
        self.beta = beta
        inner = T.optim.SGD(net.parameters(), lr=lr, momentum=beta)
        super().__init__(inner)

    def get_diagonal_preconditioner(self):
        return None


class SGDNesterovOptimizer(OptimizerWrapper):
    name = "SGD-Nesterov"

    def __init__(self, net, lr, beta=0.9):
        self.beta = beta
        inner = T.optim.SGD(net.parameters(), lr=lr, momentum=beta, nesterov=True)
        super().__init__(inner)

    def get_diagonal_preconditioner(self):
        return None


class AdamOptimizer(OptimizerWrapper):
    name = "Adam"

    def __init__(self, net, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        inner = T.optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
        super().__init__(inner)

    def get_diagonal_preconditioner(self):
        """Return flat tensor of 1/(sqrt(v_hat_i) + eps) using frozen 2nd moments.

        v_hat_i = v_i / (1 - beta2^step) is the bias-corrected second moment.
        Returns None if no optimizer state exists yet (first step).
        """
        pieces = []
        has_state = False
        for p in self.inner.param_groups[0]['params']:
            state = self.inner.state.get(p)
            if state and 'exp_avg_sq' in state:
                has_state = True
                v = state['exp_avg_sq']
                t = state['step'].item() if torch.is_tensor(state['step']) else state['step']
                bc2 = 1.0 - self.beta2 ** t
                v_hat = v / bc2
                p_diag = 1.0 / (v_hat.sqrt() + self.eps)
                pieces.append(p_diag.contiguous().view(-1))
            else:
                # No state yet — treat as P = I for this parameter
                pieces.append(torch.ones(p.numel(), device=p.device))

        if not has_state:
            return None
        return torch.cat(pieces).detach()


_REGISTRY = {
    'SGD': SGDOptimizer,
    'SGD-Momentum': SGDMomentumOptimizer,
    'SGD-Nesterov': SGDNesterovOptimizer,
    'Adam': AdamOptimizer,
}


def create_optimizer(name, net, lr, optimizer_params=None):
    """Factory: create an OptimizerWrapper by name."""
    if optimizer_params is None:
        optimizer_params = {}
    if name not in _REGISTRY:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(_REGISTRY.keys())}")
    cls = _REGISTRY[name]
    return cls(net, lr, **optimizer_params)
