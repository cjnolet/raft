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
#include "legate.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include "legate_library.h"
#include "legate_raft_cffi.h"

#include "core/cuda/stream_pool.h"
#include "core/utilities/dispatch.h"

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/handle.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/init.cuh>
#include <raft/sparse/linalg/spmm.cuh>

namespace legate_raft {

namespace {

struct sparse_csr_mm_fn_cpu {

  template <legate::Type::Code CODE>
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
    auto C_acc = C.reduce_accessor<legate::SumReduction<VAL>, true, 2>();

    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      auto i = it[0]; // row  [0, m]
      for (auto jj = Ap_acc[*it]; jj < Ap_acc[*it + 1]; ++jj) {
        auto j = Aj_acc[jj];  // [0, n]
        auto A_val = Ax_acc[jj];
        for (int32_t k = 0; k < p; ++k) {  // [0, p]
          auto B_val = B_acc[{j, k}];
          C_acc.reduce({i, k}, A_val * B_val);
        }
      }
    }
  }
};

template <legate::Type::Code CODE>
constexpr bool is_supported_gpu = (CODE == legate::Type::Code::FLOAT32 || CODE == legate::Type::Code::FLOAT64);

struct sparse_csr_mm_fn_gpu {

  template <legate::Type::Code CODE, std::enable_if_t<is_supported_gpu<CODE>>* = nullptr>
  void operator()(
      legate::Store& Ax, legate::Store& Aj, legate::Store& Ap,
      legate::Store& B, legate::Store& C,
      int32_t m, int32_t k, int32_t n,
      uint64_t nnz)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto Ax_acc = Ax.read_accessor<VAL, 1>();
    auto Aj_acc = Aj.read_accessor<int32_t, 1>();
    auto Ap_acc = Ap.read_accessor<int32_t, 1>();

    const auto shape_ = Ap.shape<1>();
    const auto shape = legate::Rect<1>(shape_.lo, shape_.hi - 1);

    auto p = B.shape<2>().hi[1] + 1;
    auto B_acc = B.read_accessor<VAL, 2>();
    auto C_acc = C.write_accessor<VAL, 2>();

    cudaStream_t stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
    raft::handle_t handle(stream);

    // Compute local partition size and offset.
    auto Ap_shape = Ap.shape<1>();
    const int32_t offset = Ap_shape.lo[0];
    const int32_t local_m = Ap_shape.hi[0] - Ap_shape.lo[0] + 1;
    const uint64_t local_nnz = std::min(static_cast<uint64_t>(local_m * k), nnz);

    // Perform local calculation if there are non-zero elements.
    if (local_nnz > 0) {

      // Create local copy of Ap partition.
      auto Ap_local = raft::make_device_vector<int32_t, int32_t>(handle, local_m);
      raft::copy(
        Ap_local.data_handle(),
        const_cast<int*>(Ap_acc.ptr(Ap_shape) + offset),
        local_m,
        stream
      );

      // Construct device_csr_matrix_view
      auto csr_structure_view = raft::make_device_compressed_structure_view<int32_t, int32_t, uint64_t>(
        Ap_local.data_handle(),
        const_cast<int*>(Aj_acc.ptr(Aj.shape<1>())),
        local_m, k, local_nnz
      );
      auto Ax_view = raft::make_device_vector_view(Ax_acc.ptr(Ax.shape<1>()), nnz);
      raft::device_span<const VAL> ax_device_span(Ax_view.data_handle(), Ax_view.size());
      raft::device_csr_matrix_view<const VAL, int, int, uint64_t> csr_matrix_view = \
        raft::make_device_csr_matrix_view(
          ax_device_span,
          csr_structure_view
      );

      // Construct views and local partition of C
      auto B_view = raft::make_device_matrix_view<const VAL, int32_t>(B_acc.ptr(B.shape<2>()), k, n);
      auto C_view = raft::make_device_matrix_view<VAL, int32_t, raft::row_major>(C_acc.ptr(C.shape<2>()), m, n);
      auto local_C = raft::make_device_matrix<VAL, int32_t>(handle, local_m, n);

      // Perform spmm calculation.
      VAL alpha = 1.0;
      VAL beta = 0.0;
      raft::sparse::linalg::spmm(
        handle, false, false, &alpha, csr_matrix_view, B_view, &beta, local_C.view()
      );

      // Copy results to output
      raft::copy(C_view.data_handle() + offset * n, local_C.data_handle(), local_C.size(), stream);
    }
    handle.sync_stream();
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported_gpu<CODE>>* = nullptr>
  void operator()(
      legate::Store& Ax, legate::Store& Aj, legate::Store& Ap,
      legate::Store& B, legate::Store& C,
      int32_t m, int32_t k, int32_t n, uint64_t nnz)
  {
    LEGATE_ABORT;
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

      legate::type_dispatch(Ax.code(), sparse_csr_mm_fn_cpu{}, Ax, Aj, Ap, B, C);
    }

    static void gpu_variant(legate::TaskContext& context)
    {
      auto& Ax = context.inputs().at(0);
      auto& Aj = context.inputs().at(1);
      auto& Ap = context.inputs().at(2);
      auto& B = context.inputs().at(3);
      auto& C = context.outputs().at(0);

      auto m = context.scalars().at(0).value<int32_t>();
      auto k = context.scalars().at(1).value<int32_t>();
      auto n = context.scalars().at(2).value<int32_t>();
      auto nnz = context.scalars().at(3).value<uint64_t>();

      legate::type_dispatch(
        Ax.code(), sparse_csr_mm_fn_gpu{},
        Ax, Aj, Ap, B, C, m, k, n, nnz
      );
    }
};

}  // namespace legate_raft

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legate_raft::SparseCSRMMTask::register_variants();
}

}  // namespace
