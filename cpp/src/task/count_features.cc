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

#include "legate_library.h"
#include "legate_raft_cffi.h"

#include "core/utilities/dispatch.h"

namespace legate_raft {

namespace {


template <legate::LegateTypeCode CODE>
constexpr bool is_supported =
  !(legate::is_floating_point<CODE>::value || legate::is_complex<CODE>::value || CODE == HALF_LT);

struct sparse_count_features_fn {

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

    legate::type_dispatch(X_data.code(), sparse_count_features_fn{}, X_data, X_rows, X_cols, labels, result, n_classes);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::SparseCountFeaturesTask::register_variants();
}

}  // namespace
