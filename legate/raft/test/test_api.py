import numpy as np
import pytest
from hypothesis import assume, example, given, note, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array

from legate.raft.core import as_array, as_store
from legate.raft.sparse import CSRStore, as_sparse_store
from legate.raft.util import broadcast_shape, promote


@given(a=array_shapes(), b=array_shapes())
@settings(max_examples=1000)
def test_broadcast(a, b):
    try:
        broadcasted_shape = broadcast_shape(a, b)
    except ValueError:
        note(f"incompatible {a} {b}")
        with pytest.raises(ValueError):
            np.ones(a) + np.ones(b)
    else:
        note(f"compatible: {a} {b} -> {broadcasted_shape}")
        assert broadcasted_shape == (np.ones(a) + np.ones(b)).shape


@st.composite
def promotable_shapes(draw):
    to_shape = draw(array_shapes(min_dims=2))

    num_dims_to_remove = draw(st.integers(min_value=1, max_value=len(to_shape) - 1))
    front = draw(st.booleans())
    if front:
        from_shape = tuple(to_shape[num_dims_to_remove:])
    else:
        from_shape = tuple(to_shape[: len(to_shape) - num_dims_to_remove])

    try:
        assume(broadcast_shape(from_shape, to_shape) == to_shape)
    except ValueError:
        assume(False)

    return from_shape, to_shape


@example(((2,), (1, 2, 3)))
@given(shapes=promotable_shapes())
@settings(max_examples=1000)
def test_promote(shapes):
    from_shape, to_shape = shapes

    try:
        broadcast_shape(from_shape, to_shape)
    except ValueError:
        pass
    else:
        promoted = list(from_shape)
        promote(from_shape, to_shape, lambda dim, size: promoted.insert(dim, size))
        assert tuple(promoted) == to_shape


def test_csr_store_from_csr_array():
    A = csr_array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 2, 3, 0, 0],
            [0, 0, 0, 4, 0],
        ]
    )

    A_store = CSRStore.from_sparse_array(A)
    assert_array_equal(A.todense(), A_store.to_sparse_array().todense())


def test_csr_matmat():
    A = csr_array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 2, 3, 0, 0],
            [0, 0, 0, 4, 0],
        ]
    )

    B = np.array([[1, 0, 0, 0], [0, 2, 3, 0], [0, 0, 0, 4], [0, 0, 0, 5], [0, 0, 0, 6]])

    C = A @ B
    C_store = as_sparse_store(A) @ as_store(B)

    assert_array_equal(C, as_array(C_store))


def test_spmm():
    A = [
        [0, 1, 0],
        [2, 0, 3],
        [0, 0, 4],
    ]

    A = csr_array(A, dtype=np.float32)

    B = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

    C = np.array([[3, 4], [17, 22], [20, 24]])
    C_lg = CSRStore.from_sparse_array(A) @ as_store(B)

    assert_array_equal(A @ B, C)
    assert_array_equal(C, as_array(C_lg))
