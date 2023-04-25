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
#include "core/utilities/typedefs.h"

namespace legate_raft {

namespace {

template <legate::LegateTypeCode SRC_TYPE, legate::LegateTypeCode DST_TYPE>
struct convert_fn {
  template <int32_t DIM>
  void operator()(legate::Store& output, legate::Store& input)
  {
    using SRC = legate::legate_type_of<SRC_TYPE>;
    using DST = legate::legate_type_of<DST_TYPE>;

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto input_acc = input.read_accessor<SRC, DIM>();
    auto output_acc = output.write_accessor<DST, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran order*/); it.valid(); ++it) {
        auto p = *it;
        output_acc[p] = static_cast<DST>(input_acc[p]);
    }
  }
};

}  // namespace

class ConvertTask : public Task<ConvertTask, CONVERT> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input  = context.inputs()[0];
    auto& output = context.outputs()[0];

    switch (input.code()) {
        case legate::LegateTypeCode::INT64_LT:
            switch(output.code()) {
                case legate::LegateTypeCode::FLOAT_LT:
                    return legate::dim_dispatch(
                        input.dim(),
                        convert_fn<legate::LegateTypeCode::INT64_LT, legate::LegateTypeCode::FLOAT_LT>{},
                        output, input
                    );
                case legate::LegateTypeCode::DOUBLE_LT:
                    return legate::dim_dispatch(
                        input.dim(),
                        convert_fn<legate::LegateTypeCode::INT64_LT, legate::LegateTypeCode::DOUBLE_LT>{},
                        output, input
                    );
                default:
                    throw(std::runtime_error("Cannot convert to that output type."));
            }
        default:
            throw(std::runtime_error("Cannot convert from this input type."));
    }
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::ConvertTask::register_variants();
}

}  // namespace
