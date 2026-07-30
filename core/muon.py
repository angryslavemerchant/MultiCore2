"""Muon (Jordan et al. 2024): momentum-SGD whose update is orthogonalised
by Newton-Schulz iteration, for the 2D hidden matrices only. Embeddings,
the LM head, and every <2D or conv parameter stay on AdamW -- Muon's
orthogonalisation is only meaningful for parameters that act as matmuls.

This is the single biggest ingredient of the NanoGPT-speedrun stack. The
implementation follows the reference repo: 5 Newton-Schulz steps with the
tuned quintic coefficients, run in bfloat16, update scaled by
sqrt(max(1, rows/cols)) so wide and tall matrices step comparably.

Under DDP every rank runs the same orthogonalisation on the same averaged
gradients -- duplicated work, identical updates. At 124M the NS cost is
noise next to a training step, so no cross-rank sharding.

ComboOptimizer merges Muon + AdamW behind the optimizer interface the
trainer already uses. Each param group carries `lr_scale`; the trainer's
schedule sets  group["lr"] = lr(step) * lr_scale  so one cosine drives both
optimizers at their own base rates.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate UV^T for G = USV^T via quintic Newton-Schulz in bf16."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = (p.grad.add(buf, alpha=group["momentum"])
                     if group["nesterov"] else buf)
                u = zeropower_via_newtonschulz5(g, group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(u.to(p.dtype), alpha=-group["lr"] * scale)


class ComboOptimizer:
    """Several optimizers behind one optimizer-shaped interface."""

    def __init__(self, opts):
        self.opts = list(opts)

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd["opts"]):
            o.load_state_dict(s)
