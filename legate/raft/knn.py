import legate.core.types as types
from legate.core import Rect
from .library import user_context as context
from .library import user_lib
from .core import array_to_store, store_to_array
import numpy as np


def run_knn(index, search, n_neighbors, metric='l2'):
    batch_size = 8

    # Setup input partitions
    index_store = array_to_store(index)
    search_store = array_to_store(search)
    n_features = search.shape[1]
    search_store = search_store.partition_by_tiling((batch_size, n_features))

    # Setup output partitions
    n_search_rows = search.shape[0]
    indices_output = np.zeros((n_search_rows, n_neighbors), dtype=np.int64)
    distances_output = np.zeros((n_search_rows, n_neighbors), dtype=np.float32)
    indices_store = array_to_store(indices_output)
    distances_store = array_to_store(distances_output)
    indices_store = indices_store.partition_by_tiling((batch_size, n_neighbors))
    distances_store = distances_store.partition_by_tiling((batch_size, n_neighbors))

    launch_shape = search_store.partition.color_shape
    task = context.create_manual_task(user_lib.cffi.RAFT_KNN_OP,
                                      launch_domain=Rect(launch_shape))
    task.add_scalar_arg(n_neighbors, types.int64)
    task.add_scalar_arg(metric, types.string)
    task.add_input(index_store)
    task.add_input(search_store)
    task.add_output(indices_store)
    task.add_output(distances_store)
    task.execute()

    indices = store_to_array(indices_store.store)
    distances = store_to_array(distances_store.store)

    return distances, indices
