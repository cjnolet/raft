#include <stdexcept>
#include <optional>

#include "../legate_raft.h"
#include "../legate_library.h"

#include <common/gpu_task_context.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/device_uvector.hpp>
#include <raft/core/handle.hpp>

#include <cuda_runtime.h>
#include <nccl.h>

namespace legate_raft {

    class MAKE_BLOBS_TASK : public Task<MAKE_BLOBS_TASK, MAKE_BLOBS> {
    public:
        static void gpu_variant(legate::TaskContext& context)
        {

            legate_raft::GPUTaskContext task_context{};

            int n_centers = context.scalars()[0].value<int>();

            auto& X = context.outputs()[0];
            auto& y = context.outputs()[1];
            auto& centers = context.inputs()[0];

            auto handle = task_context.handle();

            int n_samples = (X.shape<2>().hi[0] + 1) - X.shape<2>().lo[0];
            int n_features = X.shape<2>().hi[1] + 1;

            // The offset of the current partition from the start of the store
            // is used to obtain the pointer to the start of the partition.
            uint64_t X_offset = X.shape<2>().lo[0];
            uint64_t y_offset = y.shape<1>().lo[0];
            uint64_t centers_offset = centers.shape<2>().lo[0];

	    //printf("n_samples=%d, n_features=%d, X_offset=%ld, y_offset=%ld, centers_offset=%ld\n", n_samples, n_features, X_offset, y_offset, centers_offset);

            float* X_read = const_cast<float*>(X.write_accessor<float, 2>().ptr(Legion::DomainPoint(X_offset)));
            int* y_read = const_cast<int*>(y.write_accessor<int, 1>().ptr(Legion::DomainPoint(y_offset)));
            float* centers_read = const_cast<float*>(centers.read_accessor<float, 2>().ptr(Legion::DomainPoint(centers_offset)));

            auto X_view = raft::make_device_matrix_view<float, int>(X_read, n_samples, n_features);
            auto y_view = raft::make_device_vector_view<int, int>(y_read, n_samples);
            auto centers_view = raft::make_device_matrix_view<float, int>(centers_read, n_centers, n_features);

            raft::random::make_blobs<float, int>(handle,
                                    X_view,
                                    y_view,
                                    n_centers,
                                    std::optional(centers_view),
                                    std::nullopt,
                                    true,
                                    -10.0,
                                    10.0,
                                    X_offset);
        }
    };

}  // namespace legate_raft

namespace  // unnamed
{

    static void __attribute__((constructor)) register_tasks(void)
    {
        legate_raft::MAKE_BLOBS_TASK::register_variants();
    }

}  // namespace
