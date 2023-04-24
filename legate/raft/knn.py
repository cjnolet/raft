# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import legate.core.types as types
import numpy as np
from legate.core import Rect

from .core import as_array, as_store
from .library import user_context as context
from .library import user_lib


def run_knn(
    index: np.ndarray, search: np.ndarray, n_neighbors: int, metric: str = "l2"
):
    index_batch_size = 512
    query_batch_size = 8
    n_features = index.shape[1]

    # Setup index store
    index_store = as_store(index)
    index_store = index_store.partition_by_tiling((index_batch_size, n_features))

    # Setup search store
    search_store = as_store(search)
    search_store = search_store.partition_by_tiling((query_batch_size, n_features))

    # Setup buffer stores
    n_parts = index_store.partition.color_shape[0]
    buffer_size = n_parts * query_batch_size
    indices_buffer_array = np.zeros((buffer_size, n_neighbors), dtype=np.int64)
    distances_buffer_array = np.zeros((buffer_size, n_neighbors), dtype=np.float32)
    indices_buffer_store = as_store(indices_buffer_array)
    distances_buffer_store = as_store(distances_buffer_array)
    indices_buffer_store = indices_buffer_store.partition_by_tiling(
        (query_batch_size, n_neighbors)
    )
    distances_buffer_store = distances_buffer_store.partition_by_tiling(
        (query_batch_size, n_neighbors)
    )

    # Run KNN task
    nn_task = context.create_manual_task(
        user_lib.cffi.RAFT_KNN, launch_domain=Rect((n_parts, 1))
    )
    nn_task.add_scalar_arg(n_neighbors, types.int64)
    nn_task.add_scalar_arg(metric, types.string)
    nn_task.add_input(index_store)
    nn_task.add_input(search_store)
    nn_task.add_output(indices_buffer_store)
    nn_task.add_output(distances_buffer_store)
    nn_task.execute()

    # Gather buffer store partitions
    indices_buffer_array = as_array(indices_buffer_store.store)
    indices_buffer_array = np.array(indices_buffer_array, copy=True)
    indices_buffer_gathered = as_store(indices_buffer_array)
    distances_buffer_array = as_array(distances_buffer_store.store)
    distances_buffer_array = np.array(distances_buffer_array, copy=True)
    distances_buffer_gathered = as_store(distances_buffer_array)

    # Setup output stores
    n_search_rows = search.shape[0]
    indices_output = np.zeros((n_search_rows, n_neighbors), dtype=np.int64)
    distances_output = np.zeros((n_search_rows, n_neighbors), dtype=np.float32)
    indices_store = as_store(indices_output)
    distances_store = as_store(distances_output)

    # Run KNN merge task
    merge_task = context.create_manual_task(
        user_lib.cffi.RAFT_KNN_MERGE, launch_domain=Rect((1,))
    )
    merge_task.add_scalar_arg(query_batch_size, types.int64)
    merge_task.add_scalar_arg(n_parts, types.int64)
    merge_task.add_scalar_arg(n_neighbors, types.int64)

    merge_task.add_input(indices_buffer_gathered)
    merge_task.add_input(distances_buffer_gathered)
    merge_task.add_output(indices_store)
    merge_task.add_output(distances_store)
    merge_task.execute()

    # Produce output array
    indices = as_array(indices_store)
    distances = as_array(distances_store)

    return distances, indices
