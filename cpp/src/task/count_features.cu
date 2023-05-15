/* Copyright 2023 NVIDIA Corporation
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
 *
 */

#include <iostream>  // TODO: remove after debugging
#include <cstdint>

#include "core/cuda/stream_pool.h"
#include "core/utilities/dispatch.h"

#include "legate_library.h"
#include "../raft/raft_api.hpp"
#include "../legate_raft.h"

namespace legate_raft {

namespace {


template <legate::Type::Code CODE>
constexpr bool is_supported = legate::is_floating_point<CODE>::value;

struct sparse_count_features_fn_cpu {

  template <legate::Type::Code CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(
      legate::Store& data, legate::Store& rows, legate::Store& cols,
      legate::Store& labels, legate::Store& result)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape = data.shape<1>();

    auto data_acc = data.read_accessor<VAL, 1>();
    auto rows_acc = rows.read_accessor<int32_t, 1>();
    auto cols_acc = cols.read_accessor<int32_t, 1>();

    auto labels_acc = labels.read_accessor<int64_t, 1>();
    auto result_acc = result.reduce_accessor<legate::SumReduction<VAL>, true, 2>();

    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      auto p = *it;
      auto value = data_acc[p];
      auto row = rows_acc[p];
      auto col = cols_acc[p];
      auto label = labels_acc[row];

      result_acc.reduce(legate::Point<2>(label, col), value);
    }
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(
      legate::Store& data, legate::Store& rows, legate::Store& cols,
      legate::Store& labels, legate::Store& result)
  {
    LEGATE_ABORT;
  }

};


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

template <legate::Type::Code CODE>
constexpr bool is_supported_gpu = (CODE == legate::Type::Code::FLOAT32);

struct sparse_count_features_fn_gpu {

  template <legate::Type::Code CODE, std::enable_if_t<is_supported_gpu<CODE>>* = nullptr>
  void operator()(
      legate::Store& data, legate::Store& rows, legate::Store& cols,
      legate::Store& labels, legate::Store& result,
      uint64_t n_rows, uint64_t n_cols,
      uint64_t n_features)
  {
    using value_t = legate::legate_type_of<CODE>;
    using index_t = int32_t;

    auto shape = data.shape<1>();

    auto data_acc = data.read_accessor<value_t, 1>();
    auto rows_acc = rows.read_accessor<int32_t, 1>();
    auto cols_acc = cols.read_accessor<int32_t, 1>();
    auto labels_acc = labels.read_accessor<int64_t, 1>();
    auto result_acc = result.reduce_accessor<legate::SumReduction<value_t>, true, 2>();

    auto data_shape = data.shape<1>();
    auto offset = data_shape.lo[0];
    auto nnz = data_shape.hi[0] + 1 - offset;

    int block_size = 256;  // TODO: tune
    int num_blocks = (nnz + block_size - 1) / block_size;

    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();

    count_features_coo_kernel<<<num_blocks, block_size, 0, stream>>>(
      result_acc.ptr({0, 0}),
      rows_acc.ptr(offset),
      cols_acc.ptr(offset),
      data_acc.ptr(offset),
      nnz,
      n_rows,
      n_cols,
      labels_acc.ptr(0),
      // weights, has_weights,
      n_features,
      false
    );
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported_gpu<CODE>>* = nullptr>
  void operator()(
      legate::Store& data, legate::Store& rows, legate::Store& cols,
      legate::Store& labels, legate::Store& result,
      uint64_t n_rows, uint64_t n_cols,
      uint64_t n_features)
  {
    LEGATE_ABORT;
  }

};

}  // namespace

class SparseCountFeaturesTask : public Task<SparseCountFeaturesTask, COUNT_FEATURES> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {

    auto& X_data = context.inputs().at(0);
    auto& X_rows = context.inputs().at(1);
    auto& X_cols = context.inputs().at(2);

    auto& labels = context.inputs().at(3);
    auto& result = context.reductions().at(0);

    legate::type_dispatch(X_data.code(), sparse_count_features_fn_cpu{},
                          X_data, X_rows, X_cols, labels, result);
  }

  static void gpu_variant(legate::TaskContext& context)
  {
    auto& X_data = context.inputs().at(0);
    auto& X_rows = context.inputs().at(1);
    auto& X_cols = context.inputs().at(2);

    auto& labels = context.inputs().at(3);

    auto n_rows = context.scalars().at(1).value<uint64_t>();
    auto n_cols = context.scalars().at(2).value<uint64_t>();
    auto n_features = context.scalars().at(3).value<uint64_t>();

    auto& result = context.reductions().at(0);

    legate::type_dispatch(X_data.code(), sparse_count_features_fn_gpu{},
                          X_data, X_rows, X_cols, labels, result,
                          n_rows, n_cols, n_features);
  }

};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::SparseCountFeaturesTask::register_variants();
}

}  // namespace
