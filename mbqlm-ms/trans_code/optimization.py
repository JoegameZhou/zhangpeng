import logging
import math
from typing import Callable, Iterable, Tuple

# import torch
# from torch.optim import Optimizer
# from torch.optim.lr_scheduler import LambdaLR

import mindspore.ops as ops
from mindspore.nn import Optimizer


"""
### used torch operaters:

# torch: 			zeros_like
# nn:               Parameter
# optim: 			Optimizer, lr_scheduler.LambdaLR



### used mindspore operaters:

# mindspore: 		Parameter
# nn:				Optimizer
# ops:				ZerosLike

### lacked/useless torch operaters:

# l/u:				LambdaLR

"""



logger = logging.getLogger(__name__)


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1):

    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):


    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def linear_schedule_with_warmup_iterator(initial_lr, num_warmup_steps, num_training_steps, total_steps):
    for current_step in range(total_steps):
        if current_step >= num_training_steps:
            yield initial_lr
        elif current_step <= num_warmup_steps:
            yield initial_lr * float(current_step) / float(max(1, num_warmup_steps))
        else:
            yield initial_lr * max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):


    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):


    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# class AdamW(Optimizer):

#     def __init__(
#         self,
#         params: Iterable[torch.nn.parameter.Parameter],
#         lr: float = 1e-3,
#         betas: Tuple[float, float] = (0.9, 0.999),
#         eps: float = 1e-6,
#         weight_decay: float = 0.0,
#         correct_bias: bool = True,
#     ):
#         if lr < 0.0:
#             raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
#         super().__init__(params, defaults)

#     def step(self, closure: Callable = None):

#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

#                 state = self.state[p]
#                 if len(state) == 0:
#                     state["step"] = 0
#                     state["exp_avg"] = ops.ZerosLike()(p.data)
#                     state["exp_avg_sq"] = ops.ZerosLike()(p.data)

#                 exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
#                 beta1, beta2 = group["betas"]

#                 state["step"] += 1

#                 exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
#                 denom = exp_avg_sq.sqrt().add_(group["eps"])

#                 step_size = group["lr"]
#                 if group["correct_bias"]: 
#                     bias_correction1 = 1.0 - beta1 ** state["step"]
#                     bias_correction2 = 1.0 - beta2 ** state["step"]
#                     step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

#                 p.data.addcdiv_(exp_avg, denom, value=-step_size)

#                 if group["weight_decay"] > 0.0:
#                     p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

#         return loss