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
 */

#include <vector>

#include "legate_library.h"
#include "legate_raft_cffi.h"

#include "core/utilities/dispatch.h"

namespace legate_raft {

namespace {

struct sparse_csr_mm_fn {

  // template <legate::LegateTypeCode CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  template <legate::LegateTypeCode CODE>
  void operator()(
      legate::Store& Ax, legate::Store& Aj, legate::Store& Ap,
      legate::Store& B, legate::Store& C)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto Ax_acc = Ax.read_accessor<VAL, 1>();
    auto Aj_acc = Aj.read_accessor<int32_t, 1>();
    auto Ap_acc = Ap.read_accessor<int32_t, 1>();

    const auto shape_ = Ap.shape<1>();
    const auto shape = legate::Rect<1>(shape_.lo, shape_.hi - 1);

    auto p = B.shape<2>().hi[1] + 1;
    auto B_acc = B.read_accessor<VAL, 2>();
    auto C_acc = C.reduce_accessor<legate::SumReduction<VAL>, true, 3>();

    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      auto i = it[0]; // row  [0, m]
      for (auto jj = Ap_acc[*it]; jj < Ap_acc[*it + 1]; ++jj) {
        auto j = Aj_acc[jj];  // [0, n]
        auto A_val = Ax_acc[jj];
        for (int32_t k = 0; k < p; ++k) {  // [0, p]
          auto B_val = B_acc[{j, k}];
          C_acc.reduce({i, j, k}, A_val * B_val);
        }
      }
    }
  }
};

}  // namespace

class SparseCSRMMTask : public Task<SparseCSRMMTask, SPARSE_CSR_MM> {
  public:
    static void cpu_variant(legate::TaskContext& context)
    {
      auto& Ax = context.inputs().at(0);
      auto& Aj = context.inputs().at(1);
      auto& Ap = context.inputs().at(2);
      auto& B = context.inputs().at(3);
      auto& C = context.reductions().at(0);

      legate::type_dispatch(Ax.code(), sparse_csr_mm_fn{}, Ax, Aj, Ap, B, C);
    }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::SparseCSRMMTask::register_variants();
}

}  // namespace
