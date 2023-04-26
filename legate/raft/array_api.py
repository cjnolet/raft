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


from numbers import Number

import legate.core.types as ty
import numpy as np
import pyarrow as pa
from legate.core import Store

from legate.raft.cffi import OpCode
from legate.raft.library import user_context as context
from legate.raft.util import promote


def fill(shape, fill_value, dtype=None) -> Store:
    if dtype is None:
        try:
            dtype = pa.from_numpy_dtype(fill_value.dtype)
        except AttributeError:
            fill_value = np.asanyarray(fill_value)
            dtype = pa.from_numpy_dtype(fill_value.dtype)

    result = context.create_store(dtype, shape, optimize_scalar=(shape == tuple()))
    assert result.type == dtype

    task = context.create_auto_task(OpCode.FILL)
    task.add_output(result)
    task.add_scalar_arg(fill_value, result.type)
    task.execute()

    return result


def srange(start, stop=None, step=None, dtype=None) -> Store:
    if stop is None:
        stop = start
        start = type(stop)(0)
    if step is None:
        step = type(stop)(1)

    assert type(start) == type(stop) == type(step)

    if dtype is None:
        try:
            dtype = pa.from_numpy_dtype(stop.dtype)
        except AttributeError:
            stop = np.asanyarray(stop)
            dtype = pa.from_numpy_dtype(stop.dtype)

    size = (stop - start) // step
    shape = (size,)

    result = context.create_store(dtype, shape, optimize_scalar=(shape == (1,)))
    assert result.type == dtype

    task = context.create_auto_task(OpCode.RANGE)
    task.add_output(result)
    task.add_scalar_arg(start, result.type)
    task.add_scalar_arg(step, result.type)
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


def subtract(x1: Store | Number, x2: Store | Number) -> Store | Number:
    if isinstance(x1, Number) and isinstance(x2, Number):
        return x1 - x2  # native function
    else:
        return add(x1, negative(x2))


def max(x: Store, axis: int) -> Number:
    sanitized = _sanitize_axis(axis, x.ndim)

    try:
        limit_min = np.finfo(x.type.type.to_pandas_dtype()).min
    except ValueError:
        limit_min = np.iinfo(x.type.type.to_pandas_dtype()).min

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


def argmax(x: Store, axis: int) -> Store:
    if x.ndim != 2:
        raise NotImplementedError("argmax() only supports 2-dimensional arrays")
    if axis != 1:
        raise NotImplementedError("argmax() can only reduce on axis=1")

    # raise NotImplementedError

    result_shape = (x.shape[0],)
    result = fill(result_shape, 0, ty.int64)
    result = result.promote(1, x.shape[1])

    task = context.create_auto_task(OpCode.ARG_MAX)
    task.add_input(x)
    task.add_reduction(result, ty.ReductionOp.ADD)
    task.add_alignment(x, result)
    task.execute()

    return result.project(1, 0)


def unique(input: Store, radix: int = 4) -> Store:
    """
    Finds unique elements in the input and returns them in a store

    Parameters
    ----------
    input : Store
        Input

    Returns
    -------
    Store
        Result that contains only the unique elements of the input
    """

    if input.ndim > 1:
        raise ValueError("`unique` accepts only 1D stores")

    dtype = input.type.type
    # if num.dtype(dtype.to_pandas_dtype()).kind in ("f", "c"):
    #     raise ValueError(
    #         "`unique` doesn't support floating point or complex numbers"
    #     )

    # Create an unbound store to collect local results
    result = context.create_store(dtype, shape=None, ndim=1)

    task = context.create_auto_task(OpCode.UNIQUE)
    task.add_input(input)
    task.add_output(result)

    task.execute()

    # Perform global reduction using a reduction tree
    return context.tree_reduce(OpCode.UNIQUE, result, radix=radix)
