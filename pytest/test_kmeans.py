from legate.raft.kmeans import KMeans
import numpy as np
import sys


def test_kmeans(n_gpus, n_rows, n_cols, k):

    print("Inside test_kmeans.py", flush=True)
    X = np.random.random((n_rows*n_gpus, n_cols)).astype(np.float32)
    
    model = KMeans(n_gpus)
    model.fit(X, k)

    #print(str(model.centroids_))


if __name__ == "__main__":

    print("Inside main.", flush=True)

    n_gpus = int(sys.argv[1])
    n_rows = int(sys.argv[2])
    n_cols = int(sys.argv[3])
    k = int(sys.argv[4])

    test_kmeans(n_gpus, n_rows, n_cols, k)
