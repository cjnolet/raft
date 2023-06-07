import legate.core.types as types
from legate.core import get_legate_runtime, ingest, TiledSplit, Rect
from legate.raft.core import as_store, as_array
from legate.raft.random import make_blobs
from legate.core import get_legate_runtime     

import os, math
from legate.raft.cluster  import KMeans
import numpy as np
import sys
import time


def test_kmeans(n_rows, n_cols, k):

    print("Inside test_kmeans.py", flush=True)


    n_gpus = get_legate_runtime()._machines[0]._proc_ranges[1].high

    print("n_gpus={}".format(n_gpus))
    model = KMeans(n_gpus)

    X, y = make_blobs(n_rows, n_cols, k, n_gpus)
    get_legate_runtime().issue_execution_fence(block=True)

    fit_time = time.time()

    model.fit(X, k)

    get_legate_runtime().issue_execution_fence(block=True)

    print("Fit Took " + str(time.time() - fit_time));

    print(str(model.centroids_))


if __name__ == "__main__":


    n_rows = int(sys.argv[1])
    n_cols = int(sys.argv[2])
    k = int(sys.argv[2])

    test_kmeans(n_rows, n_cols, k)
