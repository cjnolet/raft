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

template <legate::Type::Code CODE>
constexpr bool is_supported = (legate::is_integral<CODE>::value && CODE != legate::Type::Code::BOOL);

struct map_labels_fn {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::Store& labels, legate::Store& classes, legate::Store& output)
  {
    using VAL = legate::legate_type_of<CODE>;

    const auto shape = labels.shape<1>();
    const auto classes_shape = classes.shape<1>();
    const auto n_classes = classes_shape.hi + 1;

    auto labels_acc = labels.read_accessor<VAL, 1>();
    auto classes_acc = classes.read_accessor<VAL, 1>();
    auto output_acc = output.write_accessor<VAL, 1>();

    bool found  = false;

    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      found = false;
      for (VAL idx = 0; idx < n_classes; ++idx) {
        auto label = labels_acc[*it];
        auto class_ = classes_acc[idx];
        if (label == class_) {
          output_acc[*it] = idx;
          found = true;
          break;
        }
      }
      // Failed to find label among provided classes.
      if (!found) throw std::runtime_error("Mismatch between provided labels and classes.");
    }
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(legate::Store& labels, legate::Store& classes, legate::Store& output){
    LEGATE_ABORT;
  }


};

}  // namespace

class MapLabelsTask : public Task<MapLabelsTask, MAP_LABELS> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& labels = context.inputs()[0];
    auto& classes = context.inputs()[1];
    auto& output = context.outputs()[0];

    legate::type_dispatch(labels.code(), map_labels_fn{}, labels, classes, output);
  }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::MapLabelsTask::register_variants();
}

}  // namespace
