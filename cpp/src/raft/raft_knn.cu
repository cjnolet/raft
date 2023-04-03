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


template<typename idx_t, typename value_t>
void raft_knn(idx_t n_index_rows,
              idx_t n_search_rows,
              idx_t n_features,
              idx_t k,
              std::string& metric,
              const value_t* index_ptr,
              const value_t* search_ptr,
              idx_t* indices_ptr,
              float* distances_ptr)
{
    raft::device_resources handle;

    auto index_part = raft::make_device_matrix_view<const value_t, idx_t, raft::row_major>(index_ptr, n_index_rows, n_features);
    auto search = raft::make_device_matrix_view<const value_t, idx_t, raft::row_major>(search_ptr, n_search_rows, n_features);
    auto indices = raft::make_device_matrix_view<idx_t, idx_t, raft::row_major>(indices_ptr, n_search_rows, k);
    auto distances = raft::make_device_matrix_view<value_t, idx_t, raft::row_major>(distances_ptr, n_search_rows, k);

    std::vector<raft::device_matrix_view<const value_t, idx_t, raft::row_major>> index;
    index.push_back(index_part);

    raft::distance::DistanceType distance_type;
    if (metric == "l2") {
      distance_type = raft::distance::DistanceType::L2SqrtExpanded;
    } else {
      throw std::invalid_argument("invalid metric");
    }
    raft::neighbors::brute_force::knn(handle,
                                        index,
                                        search,
                                        indices,
                                        distances,
                                        distance_type);
}


template void raft_knn(
  int64_t,
  int64_t,
  int64_t,
  int64_t,
  std::string&,
  const float*,
  const float*,
  int64_t*,
  float*);
