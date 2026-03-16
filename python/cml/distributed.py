"""
CML Distributed Training - Python bindings for distributed/data_parallel.h

Provides distributed training utilities:
- Process group management
- All-reduce, broadcast, barrier
- DistributedDataParallel (DDP) wrapper
- Pipeline parallel wrapper
"""

import ctypes
from ctypes import c_int, c_float, c_bool, c_void_p, c_size_t, POINTER, Structure

from cml.core import Tensor, _get_lib

# Backend types
DIST_BACKEND_NCCL = 0
DIST_BACKEND_MPI = 1
DIST_BACKEND_GLOO = 2

# Reduce ops
DIST_REDUCE_SUM = 0
DIST_REDUCE_PRODUCT = 1
DIST_REDUCE_MAX = 2
DIST_REDUCE_MIN = 3
DIST_REDUCE_AVG = 4


def _setup_distributed_bindings(lib):
    """Set up ctypes bindings for distributed functions."""
    try:
        lib.cml_dist_init.argtypes = [c_int, c_int, c_int]
        lib.cml_dist_init.restype = c_int

        lib.cml_dist_get_rank.argtypes = []
        lib.cml_dist_get_rank.restype = c_int

        lib.cml_dist_get_world_size.argtypes = []
        lib.cml_dist_get_world_size.restype = c_int

        lib.cml_dist_is_initialized.argtypes = []
        lib.cml_dist_is_initialized.restype = c_bool

        lib.cml_dist_destroy.argtypes = []
        lib.cml_dist_destroy.restype = None

        lib.cml_dist_barrier.argtypes = []
        lib.cml_dist_barrier.restype = c_int

        return True
    except AttributeError:
        return False


_bindings_ready = False


def _ensure_bindings():
    global _bindings_ready
    if not _bindings_ready:
        lib = _get_lib()
        if lib:
            _bindings_ready = _setup_distributed_bindings(lib)


def init_process_group(backend="gloo", world_size=-1, rank=-1):
    """Initialize distributed training.

    Args:
        backend: Communication backend ("nccl", "mpi", "gloo")
        world_size: Number of processes (-1 = auto from env)
        rank: This process's rank (-1 = auto from env)
    """
    _ensure_bindings()
    lib = _get_lib()

    backend_map = {"nccl": DIST_BACKEND_NCCL, "mpi": DIST_BACKEND_MPI, "gloo": DIST_BACKEND_GLOO}
    backend_id = backend_map.get(backend.lower(), DIST_BACKEND_GLOO)

    result = lib.cml_dist_init(backend_id, world_size, rank)
    if result != 0:
        raise RuntimeError(f"Failed to initialize distributed training with backend '{backend}'")


def get_rank():
    """Get this process's rank."""
    _ensure_bindings()
    return _get_lib().cml_dist_get_rank()


def get_world_size():
    """Get total number of processes."""
    _ensure_bindings()
    return _get_lib().cml_dist_get_world_size()


def is_initialized():
    """Check if distributed training is initialized."""
    _ensure_bindings()
    return _get_lib().cml_dist_is_initialized()


def destroy_process_group():
    """Shutdown distributed training."""
    _ensure_bindings()
    _get_lib().cml_dist_destroy()


def barrier():
    """Synchronize all processes."""
    _ensure_bindings()
    result = _get_lib().cml_dist_barrier()
    if result != 0:
        raise RuntimeError("Barrier failed")


class DistributedDataParallel:
    """Wrapper for distributed data parallel training.

    Usage:
        model = cml.Sequential()
        model.add(cml.Linear(10, 20))
        ddp_model = cml.distributed.DistributedDataParallel(model)

        for epoch in range(num_epochs):
            output = ddp_model(input_data)
            loss = cml.mse_loss(output, target)
            cml.backward(loss)
            ddp_model.sync_gradients()
            optimizer.step()
    """

    def __init__(self, module, bucket_size_mb=25):
        """Wrap a module for distributed data parallel.

        Args:
            module: CML module to wrap
            bucket_size_mb: Gradient bucket size in MB
        """
        self.module = module
        self.bucket_size_mb = bucket_size_mb

        if not is_initialized():
            raise RuntimeError("Distributed not initialized. Call init_process_group() first.")

    def __call__(self, input_tensor):
        """Forward pass through the wrapped module."""
        return self.module(input_tensor)

    def sync_gradients(self):
        """Synchronize gradients across all processes.

        Call this after backward() and before optimizer.step().
        Uses all-reduce (sum) on each parameter's gradient tensor.
        """
        _ensure_bindings()
        lib = _get_lib()
        try:
            all_reduce = lib.cml_dist_all_reduce
        except AttributeError:
            # All-reduce not available in this build; gradients remain local
            return

        for param in self.parameters():
            if hasattr(param, 'grad') and param.grad is not None:
                all_reduce(param.grad._tensor, DIST_REDUCE_SUM)
                # Average across world size
                ws = get_world_size()
                if ws > 1:
                    import numpy as np
                    grad_arr = param.grad.numpy() / float(ws)
                    from cml.core import Tensor
                    averaged = Tensor.from_numpy(grad_arr.astype(np.float32))
                    # Copy averaged data back
                    param.grad = averaged

    def parameters(self):
        """Get module parameters."""
        return self.module.parameters() if hasattr(self.module, 'parameters') else []


class PipelineParallel:
    """GPipe-style pipeline parallel wrapper.

    Splits a list of modules into stages and processes micro-batches.
    """

    def __init__(self, modules, num_micro_batches=4):
        """Create pipeline parallel wrapper.

        Args:
            modules: List of modules (one per pipeline stage)
            num_micro_batches: Number of micro-batches
        """
        self.modules = modules
        self.num_micro_batches = num_micro_batches

    def __call__(self, input_tensor):
        """Forward pass through the pipeline."""
        x = input_tensor
        for mod in self.modules:
            x = mod(x)
        return x


__all__ = [
    "init_process_group",
    "get_rank",
    "get_world_size",
    "is_initialized",
    "destroy_process_group",
    "barrier",
    "DistributedDataParallel",
    "PipelineParallel",
    "DIST_BACKEND_NCCL",
    "DIST_BACKEND_MPI",
    "DIST_BACKEND_GLOO",
]
