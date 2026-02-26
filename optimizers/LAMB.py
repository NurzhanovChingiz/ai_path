"""LAMB (Layer-wise Adaptive Moments optimizer for Batching) implementation."""
import math
from collections.abc import Callable

import torch
from torch.optim.optimizer import Optimizer, ParamsT


class Lamb(Optimizer):
    # Reference code: https://github.com/cybertronai/pytorch-lamb
    r"""Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        clamp_value (float, optional): value to clamp the trust ratio (default: 10)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
        debias (bool, optional): whether to debias the first moment estimate (default: False).

    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        clamp_value: float = 10,
        adam: bool = False,
        debias: bool = False,
    ):
        """Initialize the LAMB optimizer.

        Args:
            params: The parameters to optimize.
            lr: The learning rate.
            betas: The betas for the AdamW optimizer.
            eps: The epsilon value for the AdamW optimizer.
            weight_decay: The weight decay value for the AdamW optimizer.
            clamp_value: The clamp value for the trust ratio.
            adam: Whether to use the AdamW optimizer.
            debias: Whether to debias the first moment estimate.
        """
        if not lr >= 0.0:
            msg = f"Invalid learning rate: {lr}"
            raise ValueError(msg)
        if not eps >= 0.0:
            msg = f"Invalid epsilon value: {eps}"
            raise ValueError(msg)
        if not 0.0 <= betas[0] < 1.0:
            msg = f"Invalid beta parameter at index 0: {betas[0]}"
            raise ValueError(msg)
        if not 0.0 <= betas[1] < 1.0:
            msg = f"Invalid beta parameter at index 1: {betas[1]}"
            raise ValueError(msg)
        if weight_decay < 0:
            msg = f"Invalid weight_decay value: {weight_decay}"
            raise ValueError(msg)
        if clamp_value < 0.0:
            msg = f"Invalid clamp value: {clamp_value}"
            raise ValueError(msg)
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias

        super().__init__(params, defaults)


    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = "Lamb does not support sparse gradients, consider SparseAdam instad."
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] * \
                        math.sqrt(bias_correction2) / bias_correction1

                else:
                    step_size = group["lr"]

                weight_norm = p.data.pow(
                    2).sum().sqrt().clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(p.data, alpha=group["weight_decay"])

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss
