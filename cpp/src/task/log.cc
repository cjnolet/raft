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

struct log_fn {
  template <legate::LegateTypeCode CODE, int32_t DIM>
  void operator()(legate::Store& output, legate::Store& input)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto input_acc = input.read_accessor<VAL, DIM>();
    auto output_acc = output.write_accessor<VAL, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran order*/); it.valid(); ++it) {
        auto p = *it;
        output_acc[p] = log(input_acc[p]);
    }
  }
};

}  // namespace

class LogTask : public Task<LogTask, LOG> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input  = context.inputs()[0];
    auto& output = context.outputs()[0];

    legate::double_dispatch(input.dim(), input.code(), log_fn{}, output, input);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::LogTask::register_variants();
}

}  // namespace
