import numpy as np
from scipy.sparse import csr_array

from legate.raft.core import as_array, as_store
from legate.raft.sparse import CSRStore

A = [
    [0, 1, 0],
    [2, 0, 3],
    [0, 0, 4],
]

A = csr_array(A, dtype=np.float32)

B = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

C = np.array([[3, 4], [17, 22], [20, 24]])
C_lg = CSRStore.from_sparse_array(A) @ as_store(B)

print(C)
print(as_array(C_lg))

# assert_array_equal(A @ B, C)
# assert_array_equal(C, as_array(C_lg))
