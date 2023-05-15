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

    struct argmax_reduction_fn {

        template <legate::Type::Code CODE>
        void operator()(legate::Store& input, legate::Store& output) {

            using VAL = legate::legate_type_of<CODE>;

            const auto input_shape = input.shape<2>();
            if (input_shape.empty()) return;

            legate::Point<1> lo {0};
            legate::Point<1> hi {input_shape.hi[0]};

            const auto shape = legate::Rect<1>(lo, hi);

            auto in_acc = input.read_accessor<VAL, 2>();
            auto out_acc = output.read_write_accessor<int64_t, 2>();

            for (legate::PointInRectIterator<2> it(input_shape, false); it.valid(); ++it) {
                auto value = in_acc[*it];

                auto row = it[0];
                auto col = it[1];
                legate::Point<2> p(row, 0);
                legate::Point<2> p_max(row, out_acc[p]);

                if (value > in_acc[p_max]) {
                  out_acc[p] = col;
                }
            }
        }

    };

}

class ArgMaxTask : public Task<ArgMaxTask, ARG_MAX> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input  = context.inputs()[0];
    auto& result = context.reductions()[0];

    legate::type_dispatch(input.code(), argmax_reduction_fn{}, input, result);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::ArgMaxTask::register_variants();
}

}  // namespace
