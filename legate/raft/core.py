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

import numpy as np
import pyarrow as pa

from legate.core import Store
from legate.core._legion.future import Future

from .cffi import OpCode
from .library import user_context as context


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
