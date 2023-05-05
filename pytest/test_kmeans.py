from legate.raft.kmeans import KMeans
import numpy as np
import sys

import rmm
pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource(), initial_pool_size=2**30, maximum_pool_size=2**32)
rmm.reinitialize(pool_allocator=True, initial_pool_size=2**31)

def test_kmeans(n_gpus, n_rows, n_cols, k):
    X = np.random.random((n_rows*n_gpus, n_cols))
    
    model = KMeans(n_gpus)
    model.fit(X, k)


if __name__ == "__main__":

    n_gpus = int(sys.argv[1])
    n_rows = int(sys.argv[2])
    n_cols = int(sys.argv[3])
    k = int(sys.argv[4])

    test_kmeans(n_gpus, n_rows, n_cols, k)
