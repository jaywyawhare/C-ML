"""Optimization algorithms for training."""

from cml._cml_lib import ffi, lib


def _collect_parameters(model):
    """Collect parameters from a Module, returning (Parameter**, num_params).

    Uses module_collect_parameters(Module*, Parameter***, int*, bool recursive).
    The caller receives a pointer to the Parameter* array and the count.
    """
    params_out = ffi.new("Parameter***")
    num_params_out = ffi.new("int*")
    ret = lib.module_collect_parameters(model._module, params_out, num_params_out, True)
    if ret != 0:
        raise RuntimeError("Failed to collect parameters from model")
    return params_out[0], num_params_out[0]


class Optimizer:
    """Base optimizer wrapping a C Optimizer pointer."""

    def __init__(self, c_optimizer):
        self._optimizer = c_optimizer

    def step(self):
        lib.cml_optim_step(self._optimizer)

    def zero_grad(self):
        lib.cml_optim_zero_grad(self._optimizer)

    def set_lr(self, lr):
        lib.optimizer_set_lr(self._optimizer, float(lr))

    def get_lr(self, group_index=0):
        return lib.optimizer_get_group_lr(self._optimizer, group_index)

    def set_group_lr(self, group_index, lr):
        lib.optimizer_set_group_lr(self._optimizer, group_index, float(lr))

    def set_grad_clip_norm(self, norm):
        lib.optimizer_set_grad_clip_norm(self._optimizer, float(norm))

    def set_amsgrad(self, amsgrad):
        lib.optimizer_set_amsgrad(self._optimizer, amsgrad)

    @property
    def name(self):
        return ffi.string(lib.optimizer_get_name(self._optimizer)).decode()

    def __del__(self):
        if hasattr(self, '_optimizer') and self._optimizer != ffi.NULL:
            lib.optimizer_free(self._optimizer)


class Adam(Optimizer):
    def __init__(self, model, lr=0.001, weight_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        optimizer = lib.cml_optim_adam_for_model(
            model._module, float(lr), float(weight_decay),
            float(beta1), float(beta2), float(epsilon)
        )
        super().__init__(optimizer)


class SGD(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.0, weight_decay=0.0):
        optimizer = lib.cml_optim_sgd_for_model(
            model._module, float(lr), float(momentum), float(weight_decay)
        )
        super().__init__(optimizer)


class RMSprop(Optimizer):
    def __init__(self, model, lr=0.001, alpha=0.99, epsilon=1e-8, weight_decay=0.0):
        params, num_params = _collect_parameters(model)
        optimizer = lib.cml_optim_rmsprop(
            params, num_params, float(lr), float(weight_decay),
            float(alpha), float(epsilon)
        )
        super().__init__(optimizer)


class AdaGrad(Optimizer):
    def __init__(self, model, lr=0.01, epsilon=1e-10, weight_decay=0.0):
        params, num_params = _collect_parameters(model)
        optimizer = lib.cml_optim_adagrad(
            params, num_params, float(lr), float(weight_decay), float(epsilon)
        )
        super().__init__(optimizer)


class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)."""

    def __init__(self, model, lr=0.001, weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        params, num_params = _collect_parameters(model)
        optimizer = lib.cml_optim_adamw(
            params, num_params, float(lr), float(weight_decay),
            float(beta1), float(beta2), float(epsilon)
        )
        super().__init__(optimizer)


class NAdam(Optimizer):
    """NAdam optimizer (Adam with Nesterov momentum)."""

    def __init__(self, model, lr=0.001, weight_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        params, num_params = _collect_parameters(model)
        optimizer = lib.cml_optim_nadam(
            params, num_params, float(lr), float(weight_decay),
            float(beta1), float(beta2), float(epsilon)
        )
        super().__init__(optimizer)


class Adamax(Optimizer):
    def __init__(self, model, lr=0.002, weight_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        params, num_params = _collect_parameters(model)
        optimizer = lib.cml_optim_adamax(
            params, num_params, float(lr), float(weight_decay),
            float(beta1), float(beta2), float(epsilon)
        )
        super().__init__(optimizer)


class Adadelta(Optimizer):
    def __init__(self, model, rho=0.9, weight_decay=0.0, epsilon=1e-6):
        params, num_params = _collect_parameters(model)
        optimizer = lib.cml_optim_adadelta(
            params, num_params, float(rho), float(weight_decay), float(epsilon)
        )
        super().__init__(optimizer)


class LAMB(Optimizer):
    """LAMB optimizer (Layer-wise Adaptive Moments for Batch training)."""

    def __init__(self, model, lr=0.001, weight_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-6):
        params, num_params = _collect_parameters(model)
        optimizer = lib.cml_optim_lamb(
            params, num_params, float(lr), float(weight_decay),
            float(beta1), float(beta2), float(epsilon)
        )
        super().__init__(optimizer)


class LARS(Optimizer):
    """LARS optimizer (Layer-wise Adaptive Rate Scaling)."""

    def __init__(self, model, lr=0.1, momentum=0.9, weight_decay=0.0, trust_coefficient=0.001):
        params, num_params = _collect_parameters(model)
        optimizer = lib.cml_optim_lars(
            params, num_params, float(lr), float(momentum),
            float(weight_decay), float(trust_coefficient)
        )
        super().__init__(optimizer)


class Muon(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=False):
        params, num_params = _collect_parameters(model)
        optimizer = lib.cml_optim_muon(
            params, num_params, float(lr), float(momentum),
            float(weight_decay), nesterov
        )
        super().__init__(optimizer)


class LRScheduler:
    """Base learning rate scheduler wrapping a C LRScheduler pointer."""

    def __init__(self, c_scheduler):
        self._scheduler = c_scheduler

    def step(self, metric=0.0):
        """Update the learning rate. Returns the new learning rate.

        Args:
            metric: Current metric value (only used by ReduceOnPlateau).
        """
        return lib.cml_lr_scheduler_update(self._scheduler, float(metric))

    def get_lr(self):
        """Get the current learning rate."""
        return lib.cml_lr_scheduler_get_lr(self._scheduler)

    def __del__(self):
        if hasattr(self, '_scheduler') and self._scheduler != ffi.NULL:
            lib.cml_lr_scheduler_free(self._scheduler)


class StepLR(LRScheduler):
    """Decays the learning rate by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size, gamma=0.1):
        scheduler = lib.cml_lr_scheduler_step(
            optimizer._optimizer, int(step_size), float(gamma)
        )
        super().__init__(scheduler)


class ExponentialLR(LRScheduler):
    """Decays the learning rate by gamma every epoch."""

    def __init__(self, optimizer, gamma):
        scheduler = lib.cml_lr_scheduler_exponential(
            optimizer._optimizer, float(gamma)
        )
        super().__init__(scheduler)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate schedule."""

    def __init__(self, optimizer, T_max, eta_min=0.0):
        scheduler = lib.cml_lr_scheduler_cosine(
            optimizer._optimizer, int(T_max), float(eta_min)
        )
        super().__init__(scheduler)


class ReduceOnPlateau(LRScheduler):
    """Reduce learning rate when a metric has stopped improving."""

    def __init__(self, optimizer, factor=0.1, patience=10, min_lr=0.0):
        scheduler = lib.cml_lr_scheduler_reduce_on_plateau(
            optimizer._optimizer, float(factor), int(patience), float(min_lr)
        )
        super().__init__(scheduler)


class OneCycleLR(LRScheduler):
    """One-cycle learning rate policy."""

    def __init__(self, optimizer, max_lr, total_steps,
                 pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
        scheduler = lib.cml_lr_scheduler_one_cycle(
            optimizer._optimizer, float(max_lr), int(total_steps),
            float(pct_start), float(div_factor), float(final_div_factor)
        )
        super().__init__(scheduler)


class MultiStepLR(LRScheduler):
    """Decays the learning rate by gamma at each milestone."""

    def __init__(self, optimizer, milestones, gamma=0.1):
        milestones_arr = ffi.new("int[]", milestones)
        scheduler = lib.cml_lr_scheduler_multi_step(
            optimizer._optimizer, milestones_arr, len(milestones), float(gamma)
        )
        super().__init__(scheduler)


class PolynomialLR(LRScheduler):
    """Polynomial learning rate decay."""

    def __init__(self, optimizer, total_iters, power=1.0, min_lr=0.0):
        scheduler = lib.cml_lr_scheduler_polynomial(
            optimizer._optimizer, int(total_iters), float(power), float(min_lr)
        )
        super().__init__(scheduler)


class WarmupLR(LRScheduler):
    """Warmup wrapper around another scheduler."""

    def __init__(self, inner_scheduler, warmup_steps, warmup_start_factor=0.0):
        scheduler = lib.cml_lr_scheduler_warmup(
            inner_scheduler._scheduler, int(warmup_steps), float(warmup_start_factor)
        )
        super().__init__(scheduler)
