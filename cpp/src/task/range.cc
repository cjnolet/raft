/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this RANGEe except in compliance with the License.
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

struct range_fn {
  template <legate::LegateTypeCode CODE, int32_t DIM>
  void operator()(legate::Store& output, legate::Scalar& start, legate::Scalar& step)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape = output.shape<DIM>();

    if (shape.empty()) return;

    auto output_acc = output.write_accessor<VAL, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran order*/); it.valid(); ++it) {
        output_acc[*it] = start.value<VAL>();
        start = start.value<VAL>() + step.value<VAL>();
    }
  }
};

}  // namespace

class RangeTask : public Task<RangeTask, RANGE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& start  = context.scalars()[0];
    auto& step  = context.scalars()[1];
    auto& output = context.outputs()[0];

    legate::double_dispatch(output.dim(), output.code(), range_fn{}, output, start, step);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::RangeTask::register_variants();
}

}  // namespace
