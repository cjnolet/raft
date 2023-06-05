import legate.core.types as types
from legate.core import get_legate_runtime, ingest, TiledSplit, Rect
from legate.raft.core import as_store, as_array

import os, math
from legate.raft  import KMeans
import numpy as np
import sys
import time


def get_legate_info():
    GPUs_info = get_legate_runtime()._machines[0]._proc_ranges[1]
    legate_infos = {
        'per_node_gpus': GPUs_info.per_node_count,
        'n_workers': GPUs_info.high
    }

    if os.environ.get("OMPI_COMM_WORLD_SIZE") is not None:
        legate_infos['n_shards'] = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        legate_infos['shard_rank'] = int(os.environ["OMPI_COMM_WORLD_RANK"])
    else:
        host_to_id = {
            'dgx11': 0,
            'dgx12': 1,
            'dgx13': 2,
            'dgx14': 3,
            'dgx15': 4,
            'dgx16': 5,
            'dgx17': 6,
            'dgx18': 7,
            'dgx19': 8,
        }
        legate_infos['n_shards'] = get_legate_runtime()._num_nodes
        legate_infos['shard_rank'] = host_to_id[socket.gethostname()]

    return legate_infos

def load_dataset(dataset_dir):

    legate_infos =get_legate_info()
    def get_buffer(color):
        part_id = color[0]
        with open('{}/part_{}.npy'.format(dataset_dir, part_id), 'rb') as f:
            buf = np.load(f)
            return buf.data

    def get_local_colors():
        num_ranks = legate_infos['n_shards']
        rank = legate_infos['shard_rank']
        res = []
        i = 0
        for color in Rect((n_parts, 1)):
            if i % num_ranks == rank:
                res.append(color)
            i += 1
        return res

    base_path = os.path.basename(dataset_dir)
    split_path = base_path.split('_')

    print("base_path={}".format(split_path))
    n_rows, n_cols, n_centers, n_parts = split_path

    n_rows = int(n_rows)
    n_cols = int(n_cols)
    n_centers = int(n_centers)
    n_parts = int(n_parts)

    n_rows_per_parts = math.ceil(n_rows / n_parts)

    table = ingest(
                dtype=types.float32,
                shape=(n_rows, n_cols),
                colors=(n_parts, 1),
                data_split=TiledSplit((n_rows_per_parts, n_cols)),
                get_buffer=get_buffer,
                get_local_colors=None
            )
    data = table.__legate_data_interface__['data']
    field = next(iter(data))
    array = data[field]
    store = array.stores()[1]
    store = store.partition_by_tiling((n_rows_per_parts, n_cols))

    return store


def test_kmeans(dataset_path, k):

    print("Inside test_kmeans.py", flush=True)

    start = time.time()

    n_gpus = get_legate_runtime()._machines[0]._proc_ranges[1].high

    print("n_gpus={}".format(n_gpus))
    model = KMeans(n_gpus)

    X = load_dataset(dataset_path)

    model.fit(X, k)

    print("Took " + str(time.time() - start));


    print(str(model.centroids_))


if __name__ == "__main__":


    dataset_path = sys.argv[1]
    k = int(sys.argv[2])

    test_kmeans(dataset_path, k)
