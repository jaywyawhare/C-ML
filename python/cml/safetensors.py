"""Safetensors serialization and HuggingFace weight loading.

Implements the safetensors format (https://github.com/huggingface/safetensors)
without external dependencies: an 8-byte little-endian header length, a UTF-8
JSON header mapping names to dtype/shape/data-offsets, then one contiguous
buffer. ``save_file`` pads the header so the payload stays 8-byte aligned as
the spec requires.

``load_pretrained`` maps HuggingFace checkpoint key names onto C-ML parameter
names (e.g. ``"0.Linear.weight"``) so pretrained weights can be pulled in
without hand-written conversion scripts.

Example:
    cml.safetensors.save_file({"w": tensor}, "model.safetensors")
    state = cml.safetensors.load_file("model.safetensors")   # dict[str, Tensor]
    cml.safetensors.load_pretrained(model, "model.safetensors",
                                    mapping={"layers.0.": "0.Linear."})
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from ._cml_lib import ffi, lib
from .core import Tensor

__all__ = ["save_file", "load_file", "load_pretrained"]

_DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype(bool),
}

_NP_TO_ST = {
    "<f8": "F64",
    "<f4": "F32",
    "<i8": "I64",
    "<i4": "I32",
    "<i2": "I16",
    "i1": "I8",
    "|u1": "U8",
    "|b1": "BOOL",
}


def _st_dtype(np_dtype: np.dtype) -> str:
    key = np_dtype.str.lower()
    if key not in _NP_TO_ST:
        raise TypeError(f"unsupported safetensors dtype {np_dtype}")
    return _NP_TO_ST[key]


def save_file(tensors: dict, path: str | Path) -> None:
    """Write ``tensors`` (name -> Tensor or ndarray) to ``path``."""
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0

    for name, t in tensors.items():
        arr = t.numpy().copy() if isinstance(t, Tensor) else np.asarray(t)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        header[name] = {
            "dtype": _st_dtype(arr.dtype),
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + arr.nbytes],
        }
        blobs.append(arr.tobytes())
        offset += arr.nbytes

    # "__metadata__" is reserved by the spec; we do not emit it.
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(header_bytes)) % 8  # keep the payload 8-byte aligned
    header_bytes += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


def load_file(path: str | Path) -> dict[str, Tensor]:
    """Read a safetensors file into ``{name: Tensor}``."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        payload = f.read()

    header.pop("__metadata__", None)
    out: dict[str, Tensor] = {}
    for name, spec in sorted(header.items()):
        dtype = _DTYPES[spec["dtype"]]
        begin, end = spec["data_offsets"]
        arr = np.frombuffer(payload[begin:end], dtype=dtype).reshape(spec["shape"])
        out[name] = Tensor(arr.copy())
    return out


def load_pretrained(
    model,
    source: str | Path | dict[str, Tensor],
    mapping: dict[str, str] | None = None,
    strict: bool = False,
) -> int:
    """Load HF-style weights into a C-ML module tree.

    Args:
        model: a ``cml.nn.Module``; parameters are matched by the dotted names
            reported by ``module_collect_parameters`` (e.g.
            ``"0.Linear.weight"`` for a Sequential).
        source: path to a safetensors file or a ready ``{name: Tensor}`` dict.
        mapping: optional ``{checkpoint_prefix: parameter_prefix}`` rename
            rules applied longest-prefix-first, e.g.
            ``{"layers.0.weight": "0.Linear.weight"}``.
        strict: require every checkpoint entry to be consumed.

    Returns:
        Number of parameter tensors filled.
    """
    state = load_file(source) if isinstance(source, (str, Path)) else dict(source)

    prefixes = sorted((mapping or {}).items(),
                      key=lambda kv: len(kv[0]), reverse=True)
    params = {p.name: p for p in model.parameters(recursive=True)}

    used: set[str] = set()
    loaded = 0
    for key, value in state.items():
        target = key
        for src_prefix, dst_prefix in prefixes:
            if target.startswith(src_prefix):
                target = dst_prefix + target[len(src_prefix):]
                break
        param = params.get(target)
        if param is None:
            if strict:
                raise KeyError(f"checkpoint entry '{key}' has no matching parameter")
            continue
        src_np = value.numpy() if isinstance(value, Tensor) else np.asarray(value)
        dst_np = param.tensor.numpy()
        if dst_np.shape != src_np.shape:
            msg = (f"shape mismatch for '{key}' -> '{target}': "
                   f"{src_np.shape} vs {dst_np.shape}")
            if strict:
                raise ValueError(msg)
            continue
        _copy_into(param.tensor, src_np)
        used.add(key)
        loaded += 1

    if strict and len(used) != len(state):
        missing = sorted(set(state) - used)[:8]
        raise KeyError(f"checkpoint entries not consumed: {missing}")
    return loaded


def _remap(key: str, prefixes) -> str:
    pairs = prefixes.items() if isinstance(prefixes, dict) else prefixes
    for src, dst in sorted(pairs, key=lambda kv: len(kv[0]), reverse=True):
        if key.startswith(src):
            return dst + key[len(src):]
    return key


def _copy_into(dst: Tensor, src: np.ndarray) -> None:
    """Copy numpy data into a live parameter tensor's buffer."""
    ptr = lib.tensor_data_ptr(dst._tensor)
    if ptr == ffi.NULL:
        raise RuntimeError("parameter tensor has no data buffer")
    dst_np = dst.numpy()
    buf = ffi.buffer(ptr, dst_np.nbytes)
    buf[:] = np.ascontiguousarray(src, dtype=dst_np.dtype).tobytes()
