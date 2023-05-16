from dataclasses import dataclass
from typing import TypeAlias

from legate.core import Store
from legate.core import types as ty
from scipy.sparse import coo_array, coo_matrix, csr_array, csr_matrix

from legate.raft.core import as_store

from .array_api import fill
from .cffi import OpCode
from .core import as_array, convert
from .library import user_context as context

SparseArray: TypeAlias = csr_array | csr_matrix | coo_array | coo_matrix


@dataclass
class CSRStore:
    data: Store
    indices: Store
    indptr: Store

    shape: tuple[int]
    nnz: int

    @classmethod
    def from_array(cls, array) -> "CSRStore":
        return cls.from_sparse_array(csr_array(array))

    @classmethod
    def from_sparse_array(cls, array: SparseArray) -> "CSRStore":
        assert isinstance(array, SparseArray)
        csr = array.tocsr()
        return cls(
            data=as_store(csr.data),
            indices=as_store(csr.indices),
            indptr=as_store(csr.indptr),
            shape=csr.shape,
            nnz=csr.nnz,
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def type(self):
        return self.data.type

    def to_type(self, type_):
        return self.__class__(
            data=convert(self.data, type_),
            indices=self.indices,
            indptr=self.indptr,
            shape=self.shape,
            nnz=self.nnz,
        )

    def to_sparse_array(self) -> csr_array:
        return csr_array(
            (as_array(self.data), as_array(self.indices), as_array(self.indptr)),
            shape=self.shape,
        )

    def __matmul__(self, other: "CSRStore") -> "CSRStore":
        if isinstance(other, Store):
            return _csr_mm(self, other)
        else:
            raise NotImplementedError(
                f"Matrix multiplication for type {type(other)} is not supported."
            )


@dataclass
class COOStore:
    data: Store
    row: Store
    col: Store

    shape: tuple[int]

    @classmethod
    def from_sparse_array(cls, array: SparseArray) -> "COOStore":
        assert isinstance(array, SparseArray)
        coo = array.tocoo()
        ret = cls(
            data=as_store(coo.data),
            row=as_store(coo.row),
            col=as_store(coo.col),
            shape=coo.shape,
        )
        return ret

    def to_sparse_array(self) -> coo_array:
        return coo_array(
            (as_array(self.data), (as_array(self.row), as_array(self.col))),
            shape=self.shape,
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def type(self):
        return self.data.type


SparseStore: TypeAlias = CSRStore | COOStore


def as_sparse_store(array: SparseArray) -> SparseStore:
    return CSRStore.from_sparse_array(array)


def _csr_mm(A: CSRStore, B: Store) -> Store:
    assert A.type == B.type

    m, k = A.shape
    k_, n = B.shape
    assert k == k_
    result_shape = (m, n)

    C = fill(result_shape, 0, A.type)

    task = context.create_auto_task(OpCode.SPARSE_CSR_MM)
    task.add_input(A.data)
    # This task is only implemented for these index types.
    assert A.indptr.type == A.indices.type == ty.int32
    task.add_input(A.indices)
    task.add_input(A.indptr)

    # TODO: replace broadcasts with constraints once that's possible
    task.add_alignment(A.indices, A.data)
    task.add_broadcast(A.indices)
    task.add_broadcast(A.data)

    task.add_input(B)
    task.add_broadcast(B)

    task.add_reduction(C, ty.ReductionOp.ADD)

    task.add_scalar_arg(m, ty.int32)
    task.add_scalar_arg(k, ty.int32)
    task.add_scalar_arg(n, ty.int32)
    task.add_scalar_arg(A.nnz, ty.uint64)

    task.execute()

    return C
