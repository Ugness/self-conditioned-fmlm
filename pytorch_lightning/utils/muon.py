"""Muon optimizer (Keller Jordan's reference). Newton-Schulz5 orthogonalization
on 2D weights; AdamW (caller-built) handles biases, RMSNorm gains, and 1D
embeddings. Routing matches `optax.contrib.muon`."""

from typing import List, Tuple

import torch


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the orthogonalization of G.

    Iterates X_{k+1} = a*X_k + (b*X_k X_k^T + c*(X_k X_k^T)^2) X_k for tuned (a,b,c).
    Returns a matrix with the same shape as G, with singular values ~1.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.transpose(-1, -2)
    norm = X.norm() + 1e-7
    X = X / norm
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X.to(dtype=G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: Momentum-Orthogonalized by Newton-Schulz.

    Applies only to 2D matmul-shaped parameters. 1D/embedding/bias parameters
    should be routed to an AdamW optimizer instead (see `build_muon_param_groups`).
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters, got shape {tuple(p.shape)}."
                        " Route 1D params to AdamW via build_muon_param_groups()."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
                # Scaling factor preserves output norms across rectangular weights —
                # matches Keller Jordan's reference.
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr * scale)
        return loss


def build_muon_param_groups(
    module: torch.nn.Module,
    muon_lr_fn=None,
    adamw_lr_fn=None,
    adamw_betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.0,
) -> Tuple[List[dict], List[dict]]:
    """Split params into (Muon, AdamW) groups: 2D weights → Muon; everything
    else (biases, RMSNorm gains, 1D learned tokens) → AdamW."""
    muon_params, adamw_params = [], []
    for _, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (muon_params if p.ndim == 2 else adamw_params).append(p)
    return ([{"params": muon_params}],
            [{"params": adamw_params, "weight_decay": weight_decay, "betas": adamw_betas}])
