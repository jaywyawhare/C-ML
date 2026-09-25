"""Distributed and data-parallel training."""

from cml._cml_lib import ffi, lib
from cml.core import Tensor, _get_lib

DIST_BACKEND_NCCL = 0
DIST_BACKEND_MPI = 1
DIST_BACKEND_GLOO = 2

DIST_REDUCE_SUM = 0
DIST_REDUCE_PRODUCT = 1
DIST_REDUCE_MAX = 2
DIST_REDUCE_MIN = 3
DIST_REDUCE_AVG = 4


def init_process_group(backend="gloo", world_size=-1, rank=-1):
    backend_map = {"nccl": DIST_BACKEND_NCCL, "mpi": DIST_BACKEND_MPI, "gloo": DIST_BACKEND_GLOO}
    backend_id = backend_map.get(backend.lower(), DIST_BACKEND_GLOO)

    result = lib.cml_dist_init(backend_id, world_size, rank)
    if result != 0:
        raise RuntimeError(f"Failed to initialize distributed training with backend '{backend}'")


def get_rank():
    return lib.cml_dist_get_rank()


def get_world_size():
    return lib.cml_dist_get_world_size()


def is_initialized():
    return lib.cml_dist_is_initialized()


def destroy_process_group():
    lib.cml_dist_destroy()


def barrier():
    if lib.cml_dist_barrier() != 0:
        raise RuntimeError("Barrier failed")


class DistributedDataParallel:
    """Wraps a module in the C DDP engine (``cml_ddp_create``).

    Construction broadcasts every parameter from rank 0 and lays out the
    gradient buckets; :meth:`sync_gradients` runs the bucketed all-reduce and
    averages by world size. All of that lives in C -- this class only forwards
    the configuration and keeps the handle alive.

    ``gradient_as_bucket_view`` aliases each gradient onto its slot in the flat
    bucket instead of copying in and out around the all-reduce.
    """

    def __init__(self, module, bucket_size_mb=25, broadcast_buffers=True,
                 find_unused_parameters=False, gradient_as_bucket_view=False):
        if not is_initialized():
            raise RuntimeError("Distributed not initialized. Call init_process_group() first.")

        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self._ddp = ffi.NULL

        cfg = ffi.new("DDPConfig*")
        cfg.bucket_size_bytes = int(bucket_size_mb) * 1024 * 1024
        cfg.broadcast_buffers = bool(broadcast_buffers)
        cfg.find_unused_parameters = bool(find_unused_parameters)
        cfg.gradient_as_bucket_view = 1 if gradient_as_bucket_view else 0

        handle = lib.cml_ddp_create(ffi.cast("Module*", module._module), cfg)
        if handle == ffi.NULL:
            raise RuntimeError("cml_ddp_create failed (no parameters, or not initialized)")
        self._ddp = handle

    def __call__(self, input_tensor):
        out = lib.cml_ddp_forward(self._ddp, input_tensor._tensor)
        if out == ffi.NULL:
            raise RuntimeError("DDP forward failed")
        return Tensor(out)

    def shard_input(self, full_batch):
        """This rank's slice of a global batch along dim 0. Returns the input
        unchanged at world_size 1; feed each rank its own shard so data
        parallelism trains on distinct data rather than N identical replicas."""
        out = lib.cml_ddp_shard_input(self._ddp, full_batch._tensor)
        if out == ffi.NULL:
            raise RuntimeError("DDP input sharding failed")
        if out == full_batch._tensor:
            return full_batch
        return Tensor(out)

    def sync_gradients(self):
        """Bucketed all-reduce of every gradient, averaged by world size. Call
        after backward and before the optimizer step."""
        if lib.cml_ddp_sync_gradients(self._ddp) != 0:
            raise RuntimeError("DDP gradient sync failed")

    def parameters(self):
        return self.module.parameters() if hasattr(self.module, "parameters") else []

    def __del__(self):
        d = getattr(self, "_ddp", None)
        if d is not None and d != ffi.NULL:
            lib.cml_ddp_free(d)  # does not free the wrapped module
            self._ddp = ffi.NULL


def build_pipeline_schedule(num_stages, num_micro_batches, interleaved=False):
    """The pipeline execution order as a list of ``(stage, micro_batch, kind)``
    tuples, where kind is ``"forward"`` or ``"backward"``.

    ``interleaved`` selects 1F1B, which retires each backward as early as the
    dependencies allow so fewer micro-batches of activations are live at once;
    the default GPipe order runs every forward before any backward. Both produce
    the same gradients."""
    n_out = ffi.new("int*")
    units = lib.cml_pipeline_build_schedule(int(num_stages), int(num_micro_batches),
                                           bool(interleaved), n_out)
    if units == ffi.NULL:
        raise ValueError("invalid pipeline schedule parameters")
    try:
        return [
            (units[i].stage, units[i].micro_batch,
             "forward" if units[i].kind == lib.PIPE_UNIT_FORWARD else "backward")
            for i in range(n_out[0])
        ]
    finally:
        lib.cml_free(units)


class PipelineParallel:
    """Wraps a stage list in the C pipeline engine (``cml_pipeline_create``).

    The input batch is split into ``num_micro_batches`` micro-batches that flow
    through the stages in the configured schedule order -- GPipe by default, or
    1F1B when ``interleaved`` is set (see :func:`build_pipeline_schedule`).
    """

    def __init__(self, modules, num_micro_batches=4, interleaved=False):
        self.modules = list(modules)
        self.num_micro_batches = num_micro_batches
        self.interleaved = interleaved
        self._pipeline = ffi.NULL

        n = len(self.modules)
        if n == 0:
            raise ValueError("PipelineParallel needs at least one stage")

        stages = ffi.new("PipelineStage[]", n)
        for i, mod in enumerate(self.modules):
            stages[i].module = ffi.cast("Module*", mod._module)
            stages[i].stage_id = i
            stages[i].device_id = 0
            stages[i].device = 0  # DEVICE_CPU

        cfg = ffi.new("PipelineConfig*")
        cfg.num_micro_batches = int(num_micro_batches)
        cfg.num_stages = n
        cfg.interleaved = bool(interleaved)

        handle = lib.cml_pipeline_create(stages, n, cfg)
        if handle == ffi.NULL:
            raise RuntimeError("cml_pipeline_create failed")
        self._pipeline = handle
        # The C side copies the stage array but not the modules; keep the Python
        # wrappers alive so their Module* handles outlive the pipeline.
        self._stages_keepalive = stages

    def __call__(self, input_tensor):
        out = lib.cml_pipeline_forward(self._pipeline, input_tensor._tensor)
        if out == ffi.NULL:
            raise RuntimeError("Pipeline forward failed")
        return Tensor(out)

    def backward(self, grad_output):
        """Back-propagates every (stage, micro-batch) unit in schedule order.
        Under ``interleaved`` this consumes the forward's cached activations, so
        it cannot be run twice against one forward."""
        if lib.cml_pipeline_backward(self._pipeline, grad_output._tensor) != 0:
            raise RuntimeError("Pipeline backward failed")

    def schedule(self):
        return build_pipeline_schedule(len(self.modules), self.num_micro_batches, self.interleaved)

    def __del__(self):
        p = getattr(self, "_pipeline", None)
        if p is not None and p != ffi.NULL:
            lib.cml_pipeline_free(p)  # does not free the stage modules
            self._pipeline = ffi.NULL


__all__ = [
    "init_process_group",
    "get_rank",
    "get_world_size",
    "is_initialized",
    "destroy_process_group",
    "barrier",
    "DistributedDataParallel",
    "PipelineParallel",
    "build_pipeline_schedule",
    "DIST_BACKEND_NCCL",
    "DIST_BACKEND_MPI",
    "DIST_BACKEND_GLOO",
]
