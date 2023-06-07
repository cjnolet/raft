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

import time

import legate.core.types as types
from legate.core import Rect

from legate.raft.core import as_array
from legate.raft.library import user_context as context
from legate.raft.library import user_lib


class KMeans:
    def __init__(self, n_gpus):
        self.centroids_ = None
        self.n_gpus_ = n_gpus

    def fit(self, X_store, k: int):
        print("Inside fit!!!", flush=True)

        # Setup X store

        as_store_start = time.time()
        print("Tiling took " + str(time.time() - as_store_start))

        # Setup buffer stores
        n_parts = X_store.partition.color_shape[0]

        print("n_parts " + str(n_parts))
        # labels_buf = np.zeros((X_row_part_size*n_parts, 1), dtype=np.int32)

        centroids_store = context.create_store(types.float32, ndim=2)
        # labels_store = as_store(labels_buf).partition_by_tiling((X_row_part_size, 1))

        fit_start = time.time()

        # Run KMeans Fit task
        kmeans_fit_task = context.create_manual_task(
            user_lib.cffi.RAFT_KMEANS_FIT, launch_domain=Rect((n_parts, 1))
        )

        # NOTE: The configuration is order dependent
        kmeans_fit_task.add_scalar_arg(k, types.int32)
        kmeans_fit_task.add_input(X_store)
        # kmeans_fit_task.add_output(labels_store)
        kmeans_fit_task.add_output(centroids_store)
        # kmeans_fit_task.add_alignment(X_store, labels_store)
        kmeans_fit_task.add_nccl_communicator()
        kmeans_fit_task.execute()

        print("Fit took: " + str(time.time() - fit_start))

        print("Converting resulting centroids store")

        result_start = time.time()

        self.centroids_ = as_array(centroids_store)

        print("Results took: " + str(time.time() - result_start))
        return self
