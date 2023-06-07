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
from .core import create_matrix, create_vector, as_array, as_store
import cupy as cp

def make_blobs(n_samples, n_features, n_centers, n_parts, dtype=cp.float32, center_box=(-10.0, 10.0)):

    X = create_matrix(n_samples, n_features, dtype, n_parts)
    y = create_vector(n_samples, cp.int32, n_parts)

    make_blobs_task = context.create_manual_task(user_lib.cffi.MAKE_BLOBS,
                                                 launch_domain=Rect((n_parts, 1)))

    centers = cp.uniform(center_box[0], center_box[1], size=(n_centers, n_features)).get()
    centers_store = as_store(centers)

    # NOTE: The configuration is order dependent
    make_blobs_task.add_scalar_arg(n_centers, types.int32)
    make_blobs_task.add_input(X)
    make_blobs_task.add_input(y)
    make_blobs_task.add_broadcast(centers_store)
    make_blobs_task.execute()

    return X, y
