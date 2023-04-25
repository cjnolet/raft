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

 #include <iostream>  // TODO: remove after debugging

 #include "raft_api.hpp"

 #include <cstdint>
 #include <raft/core/device_resources.hpp>
 #include <raft/core/mdspan_types.hpp>
 #include <raft/core/device_mdspan.hpp>


 template<typename value_t, typename index_t, typename label_t>
 __global__
 void count_features_coo_kernel(value_t* out,
                                index_t* rows,
                                index_t* cols,
                                const value_t* vals,
                                int nnz,
                                int n_rows,
                                int n_cols,
                                const label_t* labels,
                                // value_t *weights,
                                // bool has_weights,
                                int n_features,
                                bool square)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nnz) return;

  index_t row = rows[i];
  index_t col = cols[i];
  value_t val = vals[i];
  label_t label = labels[row];

  auto out_idx = (label * n_features) + col;

  // if (has_weights) val *= weights[i];
  if (square) val *= val;
  atomicAdd(out + out_idx, val);
}

template __global__ void count_features_coo_kernel(
  float*, const int32_t *, const int32_t *, const float *,
  int, int, int, const int64_t *, int, bool
);


template<typename value_t, typename index_t, typename label_t>
void count_features_coo(value_t* out,
                        const index_t* rows,
                        const index_t* cols,
                        const value_t* vals,
                        int nnz,
                        int n_rows,
                        int n_cols,
                        const label_t* labels,
                        // const value_t* weights,
                        // bool has_weights,
                        int n_features,
                        bool square)

{
  int block_size = 256;  // TODO: tune
  int num_blocks = (nnz + block_size - 1) / block_size;

  count_features_coo_kernel<<<num_blocks, block_size>>>(
    out, rows, cols, vals, nnz, n_rows, n_cols, labels,
    // weights, has_weights,
    n_features, square
  );
}

template void count_features_coo(
  float*,  // value_t
  const int32_t*, // index_t
  const int32_t*, // index_t
  const float*,  // value_t
  int,
  int,
  int,
  const int64_t*, // label_t
  // const int64_t*, // value_t
  // bool,
  int,
  bool
);
