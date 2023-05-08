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

//  #include <raft/core/device_resources.hpp>

#include "../raft/raft_api.hpp"
#include "../legate_raft.h"
#include "legate_library.h"
#include "core/utilities/dispatch.h"

namespace legate_raft {

namespace {


template <legate::LegateTypeCode CODE>
constexpr bool is_supported = legate::is_floating_point<CODE>::value;

struct sparse_count_features_fn_cpu {

  template <legate::LegateTypeCode CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(
      legate::Store& data, legate::Store& rows, legate::Store& cols,
      legate::Store& labels, legate::Store& result, uint64_t n_classes)
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

  template <legate::LegateTypeCode CODE, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(
      legate::Store& data, legate::Store& rows, legate::Store& cols,
      legate::Store& labels, legate::Store& result, uint64_t n_classes)
  {
    LEGATE_ABORT;
  }

};

template <legate::LegateTypeCode CODE>
constexpr bool is_supported_gpu = (CODE == FLOAT_LT);

struct sparse_count_features_fn_gpu {

  template <legate::LegateTypeCode CODE, std::enable_if_t<is_supported_gpu<CODE>>* = nullptr>
  void operator()(
      legate::Store& data, legate::Store& rows, legate::Store& cols,
      legate::Store& labels, legate::Store& result,
      int nnz, uint64_t n_rows, uint64_t n_cols,
      uint64_t n_features, uint64_t n_classes,
      std::vector<legate::comm::Communicator>& comms)
  {
    // raft::device_resources handle;

    using VAL = legate::legate_type_of<CODE>;

    auto shape = data.shape<1>();

    auto data_acc = data.read_accessor<VAL, 1>();
    auto rows_acc = rows.read_accessor<int32_t, 1>();
    auto cols_acc = cols.read_accessor<int32_t, 1>();

    auto labels_acc = labels.read_accessor<int64_t, 1>();
    auto result_acc = result.reduce_accessor<legate::SumReduction<VAL>, true, 2>();

    auto data_shape = data.shape<1>();
    auto offset = data_shape.lo[0];
    // auto result_offset = result.shape<2>().lo[0];
    auto nnz_ = data_shape.hi[0] + 1 - offset;

    void * nccl_comm = 0;

    if (comms.size() > 0) {
      nccl_comm = comms[0].get<void*>();
    }

    count_features_coo(
      result_acc.ptr({0, 0}),
      rows_acc.ptr(offset),
      cols_acc.ptr(offset),
      data_acc.ptr(offset),
      nnz_,
      n_rows,
      n_cols,
      labels_acc.ptr(0),
      // NULL,
      // false,
      n_features,
      n_classes,
      false,
      nccl_comm
    );

  }

  template <legate::LegateTypeCode CODE, std::enable_if_t<!is_supported_gpu<CODE>>* = nullptr>
  void operator()(
      legate::Store& data, legate::Store& rows, legate::Store& cols,
      legate::Store& labels, legate::Store& result,
      int nnz, uint64_t n_rows, uint64_t n_cols,
      uint64_t n_features, uint64_t n_classes,
      std::vector<legate::comm::Communicator>& comms)
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
    auto n_classes = context.scalars().at(0).value<uint64_t>();

    auto& result = context.reductions().at(0);

    legate::type_dispatch(X_data.code(), sparse_count_features_fn_cpu{}, X_data, X_rows, X_cols, labels, result, n_classes);
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
    auto n_classes = context.scalars().at(4).value<uint64_t>();

    auto& result = context.reductions().at(0);

    auto nnz = X_rows.shape<1>().hi[0];

    legate::type_dispatch(X_data.code(), sparse_count_features_fn_gpu{},
                          X_data, X_rows, X_cols, labels, result,
                          nnz, n_rows, n_cols, n_features, n_classes,
                          context.communicators());
  }

};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::SparseCountFeaturesTask::register_variants();
}

}  // namespace
