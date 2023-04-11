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

import numpy as np
import pyarrow as pa

import legate.core.types as ty
from legate.core import Store
from legate.core._legion.future import Future

from .cffi import OpCode
from .library import user_context as context
from .util import promote


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


def array_to_store(array: np.ndarray) -> Store:
    store = context.create_store(
        pa.from_numpy_dtype(array.dtype),
        shape=array.shape,
        optimize_scalar=False,
    )
    store.attach_external_allocation(
        context,
        array.data,
        share=False,
    )
    return store


def store_to_array(store: Store) -> np.ndarray:
    if store.kind is Future:
        dtype = store.get_dtype()
        buf = store.storage.get_buffer(dtype.size)
        result = np.frombuffer(buf, dtype=dtype.type.to_pandas_dtype(), count=1)
        return result
    else:
        assert store.shape != ()

        alloc = store.get_inline_allocation(context)

        def construct_ndarray(shape, address, strides):
            dtype = np.dtype(store.get_dtype().type.to_pandas_dtype())

            initializer = _NDArray(shape, dtype.str, address, strides, False)
            result = np.asarray(initializer)
            return result

        result = alloc.consume(construct_ndarray)
        return result


def convert(input: Store, dtype: pa.DataType) -> Store:
    dtype = context.type_system[dtype]
    result = context.create_store(dtype, input.shape)
    task = context.create_auto_task(OpCode.CONVERT)
    task.add_input(input)
    task.add_output(result)
    task.add_alignment(input, result)
    task.execute()

    return result


def _sanitize_axis(axis: int, ndim: int) -> int:
    sanitized = axis
    if sanitized < 0:
        sanitized += ndim
    if sanitized < 0 or sanitized >= ndim:
        raise ValueError(f"Invalid axis {axis} for a {ndim}-D store")
    return sanitized


def sum_over_axis(input: Store, axis: int) -> Store:
    """
    Sum values along the chosen axis

    Parameters
    ----------
    input : Store
        Input to sum
    axis : int
        Axis along which the summation should be done

    Returns
    -------
    Store
        Summation result
    """
    sanitized = _sanitize_axis(axis, input.ndim)

    # Compute the output shape by removing the axis being summed over
    res_shape = tuple(ext for dim, ext in enumerate(input.shape) if dim != sanitized)
    result = fill(res_shape, 0, dtype=input.type)

    # Broadcast the output along the contracting dimension
    promoted = result.promote(axis, input.shape[axis])

    assert promoted.shape == input.shape

    task = context.create_auto_task(OpCode.SUM_OVER_AXIS)
    task.add_input(input)
    task.add_reduction(promoted, ty.ReductionOp.ADD)
    task.add_alignment(input, promoted)

    task.execute()

    return result


def multiply(rhs1: Store, rhs2: Store) -> Store:
    if rhs1.type != rhs2.type or rhs1.shape != rhs2.shape:
        raise ValueError("Stores to add must have the same type and shape")

    result = context.create_store(rhs1.type.type, rhs1.shape)

    task = context.create_auto_task(OpCode.MUL)
    task.add_input(rhs1)
    task.add_input(rhs2)
    task.add_output(result)
    task.add_alignment(result, rhs1)
    task.add_alignment(result, rhs2)

    task.execute()

    return result


def matmul(rhs1: Store, rhs2: Store) -> Store:
    """
    Performs matrix multiplication

    Parameters
    ----------
    rhs1, rhs2 : Store
        Matrices to multiply

    Returns
    -------
    Store
        Multiplication result
    """
    if rhs1.ndim != 2 or rhs2.ndim != 2:
        raise ValueError("Stores must be 2D")
    if rhs1.type != rhs2.type:
        raise ValueError("Stores must have the same type")
    if rhs1.shape[1] != rhs2.shape[0]:
        raise ValueError(
            "Can't do matrix mulplication between arrays of "
            f"shapes {rhs1.shape} and {rhs1.shape}"
        )

    m = rhs1.shape[0]
    k = rhs1.shape[1]
    n = rhs2.shape[1]

    # Multiplying an (m, k) matrix with a (k, n) matrix gives
    # an (m, n) matrix
    result = fill((m, n), 0, dtype=rhs1.type)

    # Each store gets a fake dimension that it doesn't have
    rhs1 = rhs1.promote(2, n)
    rhs2 = rhs2.promote(0, m)
    lhs = result.promote(1, k)

    assert lhs.shape == rhs1.shape
    assert lhs.shape == rhs2.shape

    task = context.create_auto_task(OpCode.MATMUL)
    task.add_input(rhs1)
    task.add_input(rhs2)
    task.add_reduction(lhs, ty.ReductionOp.ADD)
    task.add_alignment(lhs, rhs1)
    task.add_alignment(lhs, rhs2)

    task.execute()

    return result


def bincount(input: Store, num_bins: int) -> Store:
    """
    Counts the occurrences of each bin index

    Parameters
    ----------
    input : Store
        Input to bin-count
    num_bins : int
        Number of bins

    Returns
    -------
    Store
        Counting result
    """
    result = fill((num_bins,), 0, ty.uint64)

    task = context.create_auto_task(OpCode.BINCOUNT)
    task.add_input(input)
    # Broadcast the result store. This commands the Legate runtime to give
    # the entire store to every task instantiated by this task descriptor
    task.add_broadcast(result)
    # Declares that the tasks will do reductions to the result store and
    # that outputs from the tasks should be combined by addition
    task.add_reduction(result, ty.ReductionOp.ADD)

    task.execute()

    return result


def categorize(input: Store, bins: Store) -> Store:
    result = context.create_store(ty.uint64, input.shape)

    task = context.create_auto_task(OpCode.CATEGORIZE)
    task.add_input(input)
    task.add_input(bins)
    task.add_output(result)

    # Broadcast the store that contains bin edges. Each task will get a copy
    # of the entire bin edges
    task.add_broadcast(bins)

    task.execute()

    return result


def histogram(input: Store, bins: Store) -> Store:
    """
    Constructs a histogram for the given bins

    Parameters
    ----------
    input : Store
        Input
    bins : int
        Bin edges

    Returns
    -------
    Store
        Histogram
    """
    num_bins = bins.shape[0] - 1
    result = fill((num_bins,), 0, ty.uint64)

    task = context.create_auto_task(OpCode.HISTOGRAM)
    task.add_input(input)
    task.add_input(bins)
    task.add_reduction(result, ty.ReductionOp.ADD)

    # Broadcast both the result store and the one that contains bin edges.
    task.add_broadcast(bins)
    task.add_broadcast(result)

    task.execute()

    return result


def _add_constant(input: Store, value: Number) -> Store:
    result = context.create_store(input.type, input.shape)

    task = context.create_auto_task(OpCode.ADD_CONSTANT)
    task.add_input(input)
    task.add_scalar_arg(value, input.type)
    task.add_output(result)
    task.add_alignment(input, result)

    task.execute()

    return result


def log(input: Store) -> Store:
    result = context.create_store(input.type, input.shape)

    task = context.create_auto_task(OpCode.LOG)
    task.add_input(input)
    task.add_output(result)
    task.add_alignment(input, result)

    task.execute()

    return result


def exp(input: Store) -> Store:
    result = context.create_store(input.type, input.shape)

    task = context.create_auto_task(OpCode.EXP)
    task.add_input(input)
    task.add_output(result)
    task.add_alignment(input, result)

    task.execute()

    return result


def fill(shape, fill_value, dtype=None) -> Store:
    if dtype is None:
        try:
            dtype = pa.from_numpy_dtype(fill_value.dtype)
        except AttributeError:
            fill_value = np.asanyarray(fill_value)
            dtype = pa.from_numpy_dtype(fill_value.dtype)

    result = context.create_store(dtype, shape)
    assert result.type == dtype

    task = context.create_auto_task(OpCode.FILL)
    task.add_output(result)
    task.add_scalar_arg(fill_value, result.type)
    task.execute()

    return result


def add(x1: Store | Number, x2: Store | Number) -> Store | Number:
    if isinstance(x1, Number):
        if isinstance(x2, Number):
            return x1 + x2  # native function
        else:
            return add(x2, x1)  # swap operands

    elif isinstance(x2, Number):
        return _add_constant(x1, x2)
    elif x1.shape == x2.shape:
        return _add_stores(x1, x2)
    else:
        return _add_broadcast(x1, x2)


def subtract(x1: Store | Number, x2: Store | Number) -> Store | Number:
    if isinstance(x1, Number) and isinstance(x2, Number):
        return x1 - x2  # native function
    else:
        return add(x1, negative(x2))


def _add_stores(x1: Store, x2: Store) -> Store:
    result = context.create_store(x1.type, x1.shape)

    task = context.create_auto_task(OpCode.ADD)
    task.add_input(x1)
    task.add_input(x2)
    task.add_output(result)
    task.add_alignment(x1, x2)
    task.add_alignment(x1, result)

    task.execute()

    return result


def negative(lhs: Store) -> Store:
    minus_one = fill((lhs.shape), lhs.type.type.to_pandas_dtype()(-1))
    result = context.create_store(lhs.type, lhs.shape)

    task = context.create_auto_task(OpCode.MUL)
    task.add_input(lhs)
    task.add_input(minus_one)
    task.add_alignment(lhs, minus_one)
    task.add_alignment(lhs, result)
    task.add_output(result)
    task.execute()

    return result


def _add_broadcast(x1: Store, x2: Store) -> Store:
    def func(dim, dim_size):
        nonlocal x2
        x2 = x2.promote(dim, dim_size)

    promote(x2.shape, x1.shape, func)
    assert x1.shape == x2.shape

    result = context.create_store(x1.type, x1.shape)
    task = context.create_auto_task(OpCode.ADD)
    task.add_input(x1)
    task.add_input(x2)
    task.add_alignment(x1, x2)
    task.add_output(result)
    task.add_alignment(x1, result)

    task.execute()

    return result


def max(x: Store, axis: int) -> Number:
    sanitized = _sanitize_axis(axis, x.ndim)

    limit_min = np.finfo(x.type.type.to_pandas_dtype()).min
    res_shape = tuple(ext for dim, ext in enumerate(x.shape) if dim != sanitized)
    result = fill(res_shape, limit_min, x.type)

    promoted = result.promote(axis, x.shape[axis])
    assert promoted.shape == x.shape

    task = context.create_auto_task(OpCode.FIND_MAX)
    task.add_input(x)
    task.add_reduction(promoted, ty.ReductionOp.MAX)
    task.add_alignment(x, promoted)

    task.execute()

    return result
