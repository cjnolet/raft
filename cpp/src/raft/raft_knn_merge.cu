/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "raft_api.hpp"

#include <cstdint>
#include <raft/core/device_resources.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/brute_force.cuh>

#ifdef RAFT_COMPILED
#include <raft/distance/specializations.cuh>
#endif


template<typename idx_t>
void raft_knn_merge(size_t n_samples,
                    int n_parts,
                    int k,
                    const idx_t* in_ind,
                    const float* in_dist,
                    idx_t* out_ind,
                    float* out_dist)
{
    static raft::device_resources handle;

    auto in_keys = raft::device_matrix_view<const float, idx_t, raft::row_major>(in_dist, n_samples * n_parts, k);
    auto in_values = raft::device_matrix_view<const idx_t, idx_t, raft::row_major>(in_ind, n_samples * n_parts, k);
    auto out_keys = raft::device_matrix_view<float, idx_t, raft::row_major>(out_dist, n_samples, k);
    auto out_values = raft::device_matrix_view<idx_t, idx_t, raft::row_major>(out_ind, n_samples, k);

    raft::neighbors::brute_force::knn_merge_parts(handle,
                                                  in_keys,
                                                  in_values,
                                                  out_keys,
                                                  out_values,
                                                  n_samples);

}


template void raft_knn_merge(
    size_t,
    int,
    int,
    const int64_t*,
    const float*,
    int64_t*,
    float*);
