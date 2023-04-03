from dataclasses import dataclass
import pyarrow as pa
from legate.core import Store
from legate.core._legion.future import Future
from .library import user_context as context
import numpy as np


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
