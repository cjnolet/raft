import legate.core.types as types
from legate.core import get_legate_runtime, ingest, TiledSplit, Rect
from legate.raft.core import as_store, as_array
from legate.raft.datasets import load_blobs_dataset

import os, math
from legate.raft  import KMeans
import numpy as np
import sys
import time


def test_kmeans(dataset_path, k):

    print("Inside test_kmeans.py", flush=True)

    start = time.time()

    n_gpus = get_legate_runtime()._machines[0]._proc_ranges[1].high

    print("n_gpus={}".format(n_gpus))
    model = KMeans(n_gpus)

    X = load_blobs_dataset(dataset_path)

    model.fit(X, k)

    print("Took " + str(time.time() - start));


    print(str(model.centroids_))


if __name__ == "__main__":


    dataset_path = sys.argv[1]
    k = int(sys.argv[2])

    test_kmeans(dataset_path, k)
