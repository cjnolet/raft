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
from legate.core import Rect
from .library import user_context as context
from .library import user_lib
from .core import as_store, as_array
import numpy as np

import rmm
pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource(), initial_pool_size=2**30, maximum_pool_size=2**32)
rmm.reinitialize(pool_allocator=True, initial_pool_size=2**31)



class KMeans:

    def __init__(self, n_gpus):
        self.centroids_ = None
        self.n_gpus_ = n_gpus

    def fit(self, X: np.ndarray, k: int):

        # TODO: Need to figure out how to accept an existing store
        X_row_part_size =  int(X.shape[0] / self.n_gpus_)
        n_features = X.shape[1]

        # Setup X store
        X_store = as_store(X).partition_by_tiling((X_row_part_size, n_features))

        # Setup buffer stores
        n_parts = X_store.partition.color_shape[0]

        print("n_parts " + str( n_parts))
        centroids_buf = np.zeros((k*n_parts, n_features), dtype=np.float32)
        # labels_buf = np.zeros((X_row_part_size*n_parts, 1), dtype=np.int32)

        # TODO: Each task is going to end up computing this same thing individually. Need
        # to figure out how to grab it only from a single task.
        centroids_store = as_store(centroids_buf).partition_by_tiling((k, n_features))
        # labels_store = as_store(labels_buf).partition_by_tiling((X_row_part_size, 1))

        # Run KMeans Fit task
        kmeans_fit_task = context.create_manual_task(user_lib.cffi.RAFT_KMEANS_FIT,
                                                     launch_domain=Rect((n_parts, 1)))

        # NOTE: The configuration is order dependent
        kmeans_fit_task.add_scalar_arg(k, types.int32)
        kmeans_fit_task.add_input(X_store)
        # kmeans_fit_task.add_output(labels_store)
        kmeans_fit_task.add_output(centroids_store)
        # kmeans_fit_task.add_alignment(X_store, labels_store)
        kmeans_fit_task.add_nccl_communicator()
        kmeans_fit_task.execute()
        #self.centroids_ = centroids_store
        return self
