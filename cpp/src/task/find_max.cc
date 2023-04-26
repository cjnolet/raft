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
#include "core/utilities/typedefs.h"

namespace legate_raft {

namespace {

struct find_max_fn {
  template <legate::LegateTypeCode CODE, int32_t DIM>
  void operator()(legate::Store& output, legate::Store& input)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto in_acc  = input.read_accessor<VAL, DIM>();
    auto red_acc = output.reduce_accessor<legate::MaxReduction<VAL>, true, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran_order*/); it.valid(); ++it) {
      auto p = *it;
      // Coordinates of the contracting dimension are ignored by red_acc via an affine
      // transformation. For example, if the store was 3D and the second dimension was contracted,
      // each point p will go through the following affine transformation to recover the point in
      // the domain prior to the promotion:
      //
      //     | 1  0  0 |     | x |
      //     |         |  *  | y |
      //     | 0  0  1 |     | z |
      //
      // where the "*" operator denotes a matrix-vector multiplication.
      red_acc.reduce(p, in_acc[p]);
    }
  }
};


template <legate::LegateTypeCode CODE>
struct find_max_fn_outer {
  template <int32_t DIM>
  void operator()(legate::Store& output, legate::Store& input)
  {
    find_max_fn{}.operator()<CODE, DIM>(output, input);
  }
};


}  // namespace

class FindMaxTask : public Task<FindMaxTask, FIND_MAX> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input  = context.inputs()[0];
    auto& output = context.reductions()[0];

    switch(input.code()) {
      case legate::LegateTypeCode::INT8_LT:
        return legate::dim_dispatch(input.dim(), find_max_fn_outer<legate::LegateTypeCode::INT8_LT>{}, output, input);
      case legate::LegateTypeCode::INT16_LT:
        return legate::dim_dispatch(input.dim(), find_max_fn_outer<legate::LegateTypeCode::INT16_LT>{}, output, input);
      case legate::LegateTypeCode::INT32_LT:
        return legate::dim_dispatch(input.dim(), find_max_fn_outer<legate::LegateTypeCode::INT32_LT>{}, output, input);
      case legate::LegateTypeCode::INT64_LT:
        return legate::dim_dispatch(input.dim(), find_max_fn_outer<legate::LegateTypeCode::INT64_LT>{}, output, input);
      case legate::LegateTypeCode::UINT8_LT:
        return legate::dim_dispatch(input.dim(), find_max_fn_outer<legate::LegateTypeCode::UINT8_LT>{}, output, input);
      case legate::LegateTypeCode::UINT16_LT:
        return legate::dim_dispatch(input.dim(), find_max_fn_outer<legate::LegateTypeCode::UINT16_LT>{}, output, input);
      case legate::LegateTypeCode::UINT32_LT:
        return legate::dim_dispatch(input.dim(), find_max_fn_outer<legate::LegateTypeCode::UINT32_LT>{}, output, input);
      case legate::LegateTypeCode::UINT64_LT:
        return legate::dim_dispatch(input.dim(), find_max_fn_outer<legate::LegateTypeCode::UINT64_LT>{}, output, input);
      case legate::LegateTypeCode::HALF_LT:
        return legate::dim_dispatch(input.dim(), find_max_fn_outer<legate::LegateTypeCode::HALF_LT>{}, output, input);
      case legate::LegateTypeCode::FLOAT_LT:
        return legate::dim_dispatch(input.dim(), find_max_fn_outer<legate::LegateTypeCode::FLOAT_LT>{}, output, input);
      case legate::LegateTypeCode::DOUBLE_LT:
        return legate::dim_dispatch(input.dim(), find_max_fn_outer<legate::LegateTypeCode::DOUBLE_LT>{}, output, input);
      default:
        throw std::runtime_error("Input type is not supported.");
    }
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::FindMaxTask::register_variants();
}

}  // namespace
