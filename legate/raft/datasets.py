import legate.core.types as types
from legate.core import get_legate_runtime, ingest, TiledSplit, Rect
from legate.raft.core import as_store, as_array
from sklearn.datasets import make_blobs

import os, math
import numpy as np
import sys


def make_blobs_dataset_path(n_rows, n_cols, n_centers, n_parts):
    return "{}_{}_{}_{}".format(n_rows, n_cols, n_centers, n_parts)

def get_blobs_dataset_params(dataset_dir):

    base_path = os.path.basename(dataset_dir)
    split_path = base_path.split('_')

    print("base_path={}".format(split_path))
    n_rows, n_cols, n_centers, n_parts = split_path

    n_rows = int(n_rows)
    n_cols = int(n_cols)
    n_centers = int(n_centers)
    n_parts = int(n_parts)

    return n_rows, n_cols, n_centers, n_parts


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


def gen_blob_dataset(datasets_path, n_rows, n_cols, n_centers, n_parts, dtype=np.float32):
    gen_dir = '{}/{}'.format(datasets_path, make_blobs_dataset_path(n_rows, n_cols, n_centers, n_parts))
    os.mkdir(gen_dir)

    print('Generating dataset at :', gen_dir)

    X, _ = make_blobs(n_samples=n_rows,
                      n_features=n_cols,
                      centers=n_centers,
                      shuffle=True)

    blobs = X.astype(dtype)

    print("Outputting dataset parts")

    chunks = np.array_split(blobs, n_parts)
    for i, chunk in enumerate(chunks):
        with open('{}/part_{}.npy'.format(gen_dir, i), 'wb') as f:
            np.save(f, chunk)



def load_blobs_dataset(dataset_dir):

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

    n_rows, n_cols, n_centers, n_parts = get_blobs_dataset_params(dataset_dir)
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
