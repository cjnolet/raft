#include <stdexcept>

#include "../legate_raft.h"
#include "../legate_library.h"

#include <common/gpu_task_context.hpp>

#include "../raft/raft_kmeans_api.hpp"
#include <rmm/device_uvector.hpp>
#include <raft/core/handle.hpp>

#include <cuda_runtime.h>
#include <nccl.h>

namespace legate_raft {

class RAFT_KMEANS_FIT_TASK : public Task<RAFT_KMEANS_FIT_TASK, RAFT_KMEANS_FIT> {
    public:
        static void gpu_variant(legate::TaskContext& context)
        {

            legate_raft::GPUTaskContext task_context{};

            printf("Starting kmeans task\n");
            int k = context.scalars()[0].value<int>();

            printf("k=%d\n", k);

            auto& X = context.inputs()[0];

            ncclComm_t nccl_com = *(context.communicators()[0].get<ncclComm_t*>());
            task_context.inject_nccl_comm(nccl_com);

            auto handle = task_context.handle();

            int rank = handle.get_comms().get_rank();
            printf("Got NCCL comms!\n");

            int n_samples = (X.shape<2>().hi[0] + 1) - X.shape<2>().lo[0];
            int n_features = X.shape<2>().hi[1] + 1;

            printf("n_samples=%d, n_features=%d\n", n_samples, n_features);

            // The offset of the current partition from the start of the store
            // is used to obtain the pointer to the start of the partition.
            uint64_t X_offset = X.shape<2>().lo[0];
            const float* X_read = X.read_accessor<float, 2>().ptr(Legion::DomainPoint(X_offset));

            Legion::Point<2> buffer_alloc_size{k, n_features};
	    if(rank != 0) {
              buffer_alloc_size[0] = 0;
	      buffer_alloc_size[0] = 0;
	    }

	    auto& centroids = context.outputs()[0];
            auto centroids_buffer = centroids.create_output_buffer<float, 2>(buffer_alloc_size);

	    float *centroids_ptr = nullptr;


	    rmm::device_uvector<float> centroids_uvec(0, handle.get_stream());
            if(rank == 0) {
		Legion::Point<2> offset{0, 0};
		centroids_ptr = centroids_buffer.ptr(offset);
	    } else {
                centroids_uvec.resize(k * n_features, handle.get_stream());
		centroids_ptr = centroids_uvec.data();
	    }

            printf("Got centroids_buffer!\n");

            float inertia = 0;
            int n_iter;

            const float* weights = nullptr;

            printf("Invoking kmeans\n");
            kmeans::fit<float, int>(handle,
                                    k,
                                    X_read,
                                    n_samples,
                                    n_features,
                                    weights,
                                    centroids_ptr,
                                    inertia,
                                    n_iter);

            printf("Done kmeans task\n");

            printf("Returned centroids_buffer\n");


	    //if(rank == 0)

	    if(rank == 0)
		    printf("Writing bind_data\n");
	        centroids.bind_data(centroids_buffer, buffer_alloc_size);
        }
    };

}  // namespace legate_raft

namespace  // unnamed
{

    static void __attribute__((constructor)) register_tasks(void)
    {
        legate_raft::RAFT_KMEANS_FIT_TASK::register_variants();
    }

}  // namespace

