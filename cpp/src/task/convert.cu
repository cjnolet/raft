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

#include "core/cuda/stream_pool.h"
#include "core/utilities/dispatch.h"
#include "core/utilities/typedefs.h"

namespace legate_raft {

namespace {


template <legate::Type::Code SRC_TYPE, legate::Type::Code DST_TYPE>
struct convert_fn_cpu {
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


template<typename value_in_t, typename value_out_t>
__global__
void convert_kernel(const value_in_t* in, value_out_t* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = static_cast<value_out_t>(in[idx]);
}

template __global__ void convert_kernel(const int64_t*, float*);

template <legate::Type::Code SRC_TYPE, legate::Type::Code DST_TYPE>
struct convert_fn_gpu {
  template <int32_t DIM>
  void operator()(legate::Store& output, legate::Store& input)
  {
    using SRC = legate::legate_type_of<SRC_TYPE>;
    using DST = legate::legate_type_of<DST_TYPE>;

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto input_acc = input.read_accessor<SRC, DIM>();
    auto output_acc = output.write_accessor<DST, DIM>();

    auto volume = shape.volume();

    int block_size = 256;  // TODO: tune
    int num_blocks = (volume + block_size - 1) / block_size;

    // TODO: Obtain handle and stream from RMM pool.
    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();

    convert_kernel<<<num_blocks, block_size, 0, stream>>>(
        input_acc.ptr(shape), output_acc.ptr(shape)
    );
    // handle.sync_stream();
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
        case legate::Type::Code::INT64:
            switch(output.code()) {
                case legate::Type::Code::INT32:
                    return legate::dim_dispatch(
                        input.dim(),
                        convert_fn_cpu<legate::Type::Code::INT64, legate::Type::Code::INT32>{},
                        output, input
                    );
                case legate::Type::Code::FLOAT32:
                    return legate::dim_dispatch(
                        input.dim(),
                        convert_fn_cpu<legate::Type::Code::INT64, legate::Type::Code::FLOAT32>{},
                        output, input
                    );
                case legate::Type::Code::FLOAT64:
                    return legate::dim_dispatch(
                        input.dim(),
                        convert_fn_cpu<legate::Type::Code::INT64, legate::Type::Code::FLOAT64>{},
                        output, input
                    );
                default:
                    throw(std::runtime_error("Cannot convert to that output type."));
            }
        default:
            throw(std::runtime_error("Cannot convert from this input type."));
    }
  }

  static void gpu_variant(legate::TaskContext& context)
  {
    auto& input  = context.inputs()[0];
    auto& output = context.outputs()[0];

    switch (input.code()) {
        case legate::Type::Code::INT64:
            switch(output.code()) {
                case legate::Type::Code::INT32:
                    return legate::dim_dispatch(
                        input.dim(),
                        convert_fn_gpu<legate::Type::Code::INT64, legate::Type::Code::INT32>{},
                        output, input
                    );
                case legate::Type::Code::FLOAT32:
                    return legate::dim_dispatch(
                        input.dim(),
                        convert_fn_gpu<legate::Type::Code::INT64, legate::Type::Code::FLOAT32>{},
                        output, input
                    );
                case legate::Type::Code::FLOAT64:
                    return legate::dim_dispatch(
                        input.dim(),
                        convert_fn_gpu<legate::Type::Code::INT64, legate::Type::Code::FLOAT64>{},
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
