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

#include <cmath>

#include "legate_library.h"
#include "legate_raft_cffi.h"

#include "core/utilities/dispatch.h"

namespace legate_raft {

namespace {

struct add_fn {
  template <legate::LegateTypeCode CODE, int32_t DIM>
  void operator()(legate::Store& output, legate::Store& x1, legate::Store& x2)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape = x1.shape<DIM>();

    if (shape.empty()) return;

    auto x1_acc = x1.read_accessor<VAL, DIM>();
    auto x2_acc = x2.read_accessor<VAL, DIM>();
    auto output_acc = output.write_accessor<VAL, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran order*/); it.valid(); ++it) {
        auto p = *it;
        output_acc[p] = x1_acc[p] + x2_acc[p];
    }
  }
};

}  // namespace

class AddTask : public Task<AddTask, ADD> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input1  = context.inputs()[0];
    auto& input2  = context.inputs()[1];
    auto& output = context.outputs()[0];

    legate::double_dispatch(input1.dim(), input1.code(), add_fn{}, output, input1, input2);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::AddTask::register_variants();
}

}  // namespace
