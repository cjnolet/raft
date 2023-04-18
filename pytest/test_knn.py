import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from legate.raft import run_knn


def test_knn():
    k = 8
    metric = 'l2'
    n_features = 20
    n_index_rows = 500
    n_search_rows = 16

    X, _ = make_blobs(n_samples=n_index_rows + n_search_rows,
                      centers=5, n_features=n_features)
    blob_index = X[:n_index_rows].astype(np.float32)
    blob_search = X[n_index_rows:].astype(np.float32)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(blob_index)
    ref_distances, ref_indices = nn.kneighbors(blob_search, return_distance=True)

    distances, indices = run_knn(blob_index, blob_search, k, metric)
    np.testing.assert_allclose(indices, ref_indices)
    np.testing.assert_allclose(distances, ref_distances, rtol=0.001)
