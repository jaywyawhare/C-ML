"""The Python distributed wrappers must drive the C engine, not reimplement it.

DistributedDataParallel and PipelineParallel used to be Python-only shims: DDP
never called ``cml_ddp_create`` (so no rank-0 broadcast, no bucketing, and
neither ``find_unused_parameters`` nor ``gradient_as_bucket_view`` was reachable)
and PipelineParallel just chained the modules, ignoring ``num_micro_batches``
entirely. These tests pin the wrappers to the real handles.

Run with:  cd python && python3 -m pytest tests/test_distributed_py.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CML_LOG_LEVEL", "ERROR")

import numpy as np
import pytest

import cml
from cml import nn
from cml import distributed as dist
from cml._cml_lib import ffi


@pytest.fixture
def process_group():
    dist.init_process_group("gloo", 1, 0)
    yield
    dist.destroy_process_group()
    cml.reset_graph()


def _x(rows=2, cols=4):
    return cml.Tensor(np.arange(rows * cols, dtype=np.float32).reshape(rows, cols) * 0.25)


def test_process_group_lifecycle(process_group):
    assert dist.is_initialized()
    assert dist.get_rank() == 0
    assert dist.get_world_size() == 1
    dist.barrier()


def test_ddp_holds_a_real_c_handle(process_group):
    ddp = dist.DistributedDataParallel(nn.Linear(4, 3))
    assert ddp._ddp != ffi.NULL


def test_ddp_requires_process_group():
    assert not dist.is_initialized()
    with pytest.raises(RuntimeError):
        dist.DistributedDataParallel(nn.Linear(4, 3))


def test_ddp_forward_matches_plain_module(process_group):
    lin = nn.Linear(4, 3)
    x = _x()
    plain = lin(x).numpy()

    ddp = dist.DistributedDataParallel(lin)
    through_ddp = ddp(x).numpy()
    np.testing.assert_allclose(through_ddp, plain, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bucket_view", [False, True])
def test_ddp_sync_gradients_runs(process_group, bucket_view):
    """Both the copying and the bucket-view paths must complete. At world_size 1
    there is nothing to reduce, but bucket-view still rebinds the gradients onto
    their bucket slots, so this exercises that aliasing."""
    lin = nn.Linear(4, 3)
    ddp = dist.DistributedDataParallel(lin, gradient_as_bucket_view=bucket_view)
    out = ddp(_x())
    out.sum().backward()
    ddp.sync_gradients()  # raises on a non-zero return


def test_ddp_find_unused_parameters_accepted(process_group):
    ddp = dist.DistributedDataParallel(nn.Linear(4, 3), find_unused_parameters=True)
    ddp.sync_gradients()


def test_ddp_shard_input_is_identity_at_world_size_one(process_group):
    ddp = dist.DistributedDataParallel(nn.Linear(4, 3))
    x = _x(rows=4)
    assert ddp.shard_input(x) is x


def test_ddp_exposes_parameters(process_group):
    lin = nn.Linear(4, 3)
    ddp = dist.DistributedDataParallel(lin)
    assert len(ddp.parameters()) == len(lin.parameters())


# ---- pipeline schedules ----------------------------------------------------


def _is_valid_schedule(units, P, M):
    """Every unit appears once and only after what it depends on."""
    assert len(units) == 2 * P * M
    f_done, b_done = set(), set()
    for stage, mb, kind in units:
        assert 0 <= stage < P and 0 <= mb < M
        if kind == "forward":
            assert (stage, mb) not in f_done
            if stage > 0:
                assert (stage - 1, mb) in f_done
            f_done.add((stage, mb))
        else:
            assert (stage, mb) not in b_done
            assert (stage, mb) in f_done
            if stage < P - 1:
                assert (stage + 1, mb) in b_done
            b_done.add((stage, mb))
    assert len(f_done) == len(b_done) == P * M
    return True


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("P,M", [(1, 1), (2, 3), (3, 4), (4, 8)])
def test_build_pipeline_schedule_is_valid(P, M, interleaved):
    assert _is_valid_schedule(dist.build_pipeline_schedule(P, M, interleaved), P, M)


def test_gpipe_and_1f1b_orders_differ():
    g = dist.build_pipeline_schedule(3, 4, interleaved=False)
    i = dist.build_pipeline_schedule(3, 4, interleaved=True)
    assert len(g) == len(i)
    assert g != i


def test_1f1b_bounds_live_activations():
    """GPipe holds every micro-batch at once; 1F1B must hold fewer. This is the
    property the schedule exists for."""
    P, M = 4, 8

    def peak_live(units):
        live, peak = set(), 0
        for stage, mb, kind in units:
            if kind == "forward":
                live.add(mb)
                peak = max(peak, len(live))
            elif stage == 0:
                live.discard(mb)
        return peak

    peak_g = peak_live(dist.build_pipeline_schedule(P, M, False))
    peak_i = peak_live(dist.build_pipeline_schedule(P, M, True))
    assert peak_g == M
    assert peak_i <= P
    assert peak_i < peak_g


def test_build_pipeline_schedule_rejects_bad_input():
    with pytest.raises(ValueError):
        dist.build_pipeline_schedule(0, 4)
    with pytest.raises(ValueError):
        dist.build_pipeline_schedule(4, 0)


# ---- pipeline execution ---------------------------------------------------


@pytest.mark.parametrize("interleaved", [False, True])
def test_pipeline_forward_and_backward(interleaved):
    pp = dist.PipelineParallel(
        [nn.Linear(4, 4), nn.Linear(4, 2)], num_micro_batches=2, interleaved=interleaved
    )
    assert pp._pipeline != ffi.NULL

    out = pp(_x(rows=4))
    assert out.shape == (4, 2)

    pp.backward(cml.Tensor(np.ones((4, 2), dtype=np.float32)))
    assert len(pp.schedule()) == 2 * 2 * 2
    cml.reset_graph()


def test_pipeline_rejects_empty_stage_list():
    with pytest.raises(ValueError):
        dist.PipelineParallel([])


def test_pipeline_matches_sequential_composition():
    """One micro-batch through two stages is just the two modules composed."""
    a, b = nn.Linear(4, 4), nn.Linear(4, 2)
    x = _x(rows=2)
    reference = b(a(x)).numpy()

    pp = dist.PipelineParallel([a, b], num_micro_batches=1)
    np.testing.assert_allclose(pp(x).numpy(), reference, rtol=1e-5, atol=1e-5)
    cml.reset_graph()
