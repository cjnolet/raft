# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dataclasses import dataclass
from numbers import Number
from typing import TypeAlias

import numpy as np
from legate.core import Store
from legate.core import types as ty
from legate.core._legion.future import Future

from .cffi import OpCode
from .library import user_context as context

_NP2LT_TYPES = {
    np.dtype(np.bool_): ty.bool_,
    np.dtype(np.int8): ty.int8,
    np.dtype(np.int16): ty.int16,
    np.dtype(np.int32): ty.int32,
    np.dtype(np.int64): ty.int64,
    np.dtype(np.uint8): ty.uint8,
    np.dtype(np.uint16): ty.uint16,
    np.dtype(np.uint32): ty.uint32,
    np.dtype(np.uint64): ty.uint64,
    np.dtype(np.float16): ty.float16,
    np.dtype(np.float32): ty.float32,
    np.dtype(np.float64): ty.float64,
    np.dtype(np.complex64): ty.complex64,
    np.dtype(np.complex128): ty.complex128,
    np.dtype(np.str_): ty.string,
}


@dataclass
class _NDArray:
    shape: tuple[int]
    typestr: str
    ptr: int
    strides: tuple[int] | None
    read_only: bool
    version: int = 3

    @property
    def __array_interface__(self):
        return {
            "version": self.version,
            "shape": self.shape,
            "typestr": self.typestr,
            "data": (self.ptr, self.read_only),
            "strides": self.strides,
        }


def as_store(array: np.ndarray) -> Store:
    store = context.create_store(
        _NP2LT_TYPES[array.dtype],
        shape=array.shape,
        optimize_scalar=False,
    )
    store.attach_external_allocation(
        context,
        array.data,
        share=False,
    )
    return store


def as_array(store: Store) -> np.ndarray:
    if store.kind is Future:
        dtype = store.get_dtype()
        buf = store.storage.get_buffer(dtype.size)
        result = np.frombuffer(buf, dtype=dtype.type.to_pandas_dtype(), count=1)
        return result
    else:
        assert store.shape != ()

        alloc = store.get_inline_allocation(context)

        def construct_ndarray(shape, address, strides):
            dtype = store.type.to_numpy_dtype()

            initializer = _NDArray(shape, dtype.str, address, strides, False)
            result = np.asarray(initializer)
            return result

        result = alloc.consume(construct_ndarray)
        return result


def as_scalar(store: Store) -> Number:
    array = as_array(store)
    assert array.ndim == 1 and array.shape == (1,)
    return array.item()


_NativeLegateType: TypeAlias = ty.Dtype
DataType: TypeAlias = type | np.dtype | _NativeLegateType


def _determine_dtype(dtype: DataType) -> ty.Dtype:
    if type(dtype) is ty.Dtype:
        return dtype
    elif dtype is int:
        return ty.int64
    elif dtype is float:
        return ty.float64
    elif dtype is bool:
        return ty.bool_

    raise ValueError(f"Unsupported dtype: {dtype} ({type(dtype)})")


def convert(input: Store, dtype: DataType) -> Store:
    target_dtype = _determine_dtype(dtype)
    if _determine_dtype(input.type) == target_dtype:
        return input

    result = context.create_store(target_dtype, input.shape)
    task = context.create_auto_task(OpCode.CONVERT)
    task.add_input(input)
    task.add_output(result)
    task.add_alignment(input, result)
    task.execute()

    # TODO: This should not be necessary once we pro-actively sync the cuda
    # stream.
    context.issue_execution_fence()

    return result


def to_scalar(input: Store) -> Number:
    """Extracts a Python scalar value from a Legate store
       encapsulating a single scalar

    Args:
        input (Store): The Legate store encapsulating a scalar

    Returns:
        number: A Python scalar
    """
    # This operation blocks until the data in the Store
    # is available and correct
    return as_array(input)[0]
