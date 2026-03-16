"""Optimization algorithms for training."""

from cml._cml_lib import ffi, lib


class Optimizer:
    """Base class for optimizers."""

    def __init__(self, c_optimizer):
        self._optimizer = c_optimizer

    def step(self):
        lib.cml_optim_step(self._optimizer)

    def zero_grad(self):
        lib.cml_optim_zero_grad(self._optimizer)

    def set_lr(self, lr):
        lib.cml_optim_set_lr(self._optimizer, lr)

    def __del__(self):
        if self._optimizer != ffi.NULL:
            lib.optimizer_free(self._optimizer)


class Adam(Optimizer):
    def __init__(
        self, model, lr=0.001, weight_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        optimizer = lib.cml_optim_adam_for_model(
            model._module, lr, weight_decay, beta1, beta2, epsilon
        )
        super().__init__(optimizer)


class SGD(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.0, weight_decay=0.0):
        optimizer = lib.cml_optim_sgd_for_model(
            model._module, lr, momentum, weight_decay
        )
        super().__init__(optimizer)


class RMSprop(Optimizer):
    def __init__(self, model, lr=0.001, alpha=0.99, epsilon=1e-8, weight_decay=0.0):
        optimizer = lib.cml_optim_rmsprop_for_model(
            model._module, lr, alpha, epsilon, weight_decay
        )
        super().__init__(optimizer)


class AdaGrad(Optimizer):
    def __init__(self, model, lr=0.01, epsilon=1e-10):
        optimizer = lib.cml_optim_adagrad_for_model(model._module, lr, epsilon)
        super().__init__(optimizer)
